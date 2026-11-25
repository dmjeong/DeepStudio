// Fixed training-size dense inference; all workspaces are local to this call.
// Schema 8 matches python/dense_deployment.py. No session mutation/downloads.
#pragma once

namespace {
// Desktop class masks use Pillow NEAREST. OpenCV's integer/fixed-point
// nearest-even implementation can choose the adjacent pixel at exact ties.
// Match Pillow's double-precision coordinate recurrence, including its order.
std::vector<int> PreviewNearestIndices(int source, int target)
{
    const double step = static_cast<double>(source) / target;
    volatile double position = step * .5; // retain sequential rounding under MSVC optimization
    std::vector<int> indices(target);
    for (int i = 0; i < target; ++i) {
        indices[i] = std::min(source - 1, static_cast<int>(position));
        position = position + step;
    }
    return indices;
}

cv::Mat ResizePreviewNearest(const cv::Mat& source, cv::Size size)
{
    if (source.size() == size) return source;
    const auto xs = PreviewNearestIndices(source.cols, size.width);
    const auto ys = PreviewNearestIndices(source.rows, size.height);
    cv::Mat output(size, source.type());
    for (int y = 0; y < size.height; ++y) {
        if (source.type() == CV_8UC1) {
            const auto* input = source.ptr<uchar>(ys[y]);
            auto* row = output.ptr<uchar>(y);
            for (int x = 0; x < size.width; ++x) row[x] = input[xs[x]];
        } else if (source.type() == CV_32FC1) {
            const auto* input = source.ptr<float>(ys[y]);
            auto* row = output.ptr<float>(y);
            for (int x = 0; x < size.width; ++x) row[x] = input[xs[x]];
        } else throw std::invalid_argument("Unsupported preview mask type.");
    }
    return output;
}

struct DenseOutput {
    std::vector<float> values;
    int height = 0, width = 0;
    double preprocess_ms = 0, model_ms = 0, postprocess_ms = 0;
};

DenseOutput DenseTiles(const cv::Mat& region, const InferenceConfig& config,
                       Ort::Session& session, bool segmentation)
{
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b-a).count(); };
    auto started = Clock::now();
    if (region.empty() || region.depth() != CV_8U ||
        (region.channels()!=1 && region.channels()!=3 && region.channels()!=4))
        throw std::invalid_argument("Dense tile input must be uint8 gray/BGR/BGRA.");
    cv::Mat rgb;
    if (config.input_channels==1) {
        if (region.channels()==1) rgb=region;
        else cv::cvtColor(region,rgb,region.channels()==3?cv::COLOR_BGR2GRAY:cv::COLOR_BGRA2GRAY);
    } else cv::cvtColor(region,rgb,region.channels()==1?cv::COLOR_GRAY2RGB:
                       region.channels()==3?cv::COLOR_BGR2RGB:cv::COLOR_BGRA2RGB);
    DenseOutput result;
    const auto scaled = [](int size, double scale) {
        const double value = size * scale, lower = std::floor(value), fraction = value - lower;
        const int base = static_cast<int>(lower);
        return std::max(1, base + ((fraction > .5 || (fraction == .5 && base % 2)) ? 1 : 0));
    };
    result.width=scaled(region.cols,config.dense_scale);
    result.height=scaled(region.rows,config.dense_scale);
    if (rgb.cols!=result.width || rgb.rows!=result.height)
        cv::resize(rgb,rgb,cv::Size(result.width,result.height),0,0,config.dense_interpolation);
    const int tile=config.dense_tile, overlap=config.dense_overlap, margin=overlap/2;
    const int ph=std::max(tile,result.height), pw=std::max(tile,result.width);
    const size_t plane=static_cast<size_t>(ph)*pw, tp=static_cast<size_t>(tile)*tile;
    const int classes=segmentation?config.num_classes:1;
    auto pixels=vision_preprocess::normalize_padded_u8(rgb,pw,ph,config.normalize_mean,config.normalize_std);
    std::vector<float> sums(static_cast<size_t>(classes)*plane,0), weights(plane,0);
    std::vector<float> input(static_cast<size_t>(config.input_channels)*tp);
    auto axis=[tile,overlap](int length) {
        std::vector<int> positions;
        for (int p=0;p<=length-tile;p+=tile-overlap) positions.push_back(p);
        if (positions.back()!=length-tile) positions.push_back(length-tile);
        return positions;
    };
    auto edge=[tile,margin](int start,int index,int length) {
        return ((start>0 && index<margin) || (start+tile<length && index>=tile-margin))?1e-4f:1.f;
    };
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    const std::vector<int64_t> shape{1,config.input_channels,tile,tile};
    auto tensor=Ort::Value::CreateTensor<float>(memory,input.data(),input.size(),shape.data(),shape.size());
    const char* input_names[]={config.input_name.c_str()};
    const char* output_names[]={segmentation?config.output_name.c_str():config.output_names.at(1).c_str()};
    result.preprocess_ms+=ms(started,Clock::now());
    for (int y:axis(ph)) for (int x:axis(pw)) {
        auto begin=Clock::now();
        for (int c=0;c<config.input_channels;++c) for (int row=0;row<tile;++row)
            std::copy_n(pixels.data()+c*plane+static_cast<size_t>(y+row)*pw+x,tile,
                        input.data()+c*tp+static_cast<size_t>(row)*tile);
        auto model_start=Clock::now();
        auto outputs=session.Run(Ort::RunOptions{nullptr},input_names,&tensor,1,output_names,1);
        auto model_end=Clock::now();
        const auto info=outputs[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape()!=std::vector<int64_t>{1,classes,tile,tile})
            throw std::runtime_error("Dense tile output shape mismatch.");
        const float* data=outputs[0].GetTensorData<float>();
        std::vector<float> probabilities(classes);
        for (int row=0;row<tile;++row) for (int col=0;col<tile;++col) {
            const size_t local=static_cast<size_t>(row)*tile+col, global=static_cast<size_t>(y+row)*pw+x+col;
            const float weight=edge(y,row,ph)*edge(x,col,pw);
            float maximum=-std::numeric_limits<float>::infinity();
            for (int c=0;c<classes;++c) {
                const float value=data[c*tp+local];
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite dense output.");
                maximum=std::max(maximum,value);
            }
            float total=0;
            for (int c=0;c<classes;++c) {
                probabilities[c]=segmentation?std::exp(data[c*tp+local]-maximum):data[c*tp+local];
                total+=probabilities[c];
            }
            for (int c=0;c<classes;++c) sums[c*plane+global]+=weight*probabilities[c]/(segmentation?total:1.f);
            weights[global]+=weight;
        }
        result.preprocess_ms+=ms(begin,model_start);
        result.model_ms+=ms(model_start,model_end);
        result.postprocess_ms+=ms(model_end,Clock::now());
    }
    auto begin=Clock::now();
    const size_t rp=static_cast<size_t>(result.height)*result.width;
    result.values.resize(classes*rp);
    for (int c=0;c<classes;++c) for (int y=0;y<result.height;++y) for (int x=0;x<result.width;++x) {
        const size_t index=static_cast<size_t>(y)*pw+x;
        result.values[c*rp+static_cast<size_t>(y)*result.width+x]=sums[c*plane+index]/weights[index];
    }
    result.postprocess_ms+=ms(begin,Clock::now());
    return result;
}
} // namespace
