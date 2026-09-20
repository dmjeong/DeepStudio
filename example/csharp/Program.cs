using System.Diagnostics;
using System.Globalization;
using DeepVisionStudio;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
try
{
    bool selfTest = args.Length == 2 && args[0] == "--self-test";
    if (!selfTest && (args.Length < 5 || args.Length > 6))
    {
        Console.Error.WriteLine("Usage: OnnxExample <model.json> <image.raw> <width> <height> <channels> [runs=20]");
        Console.Error.WriteLine("       OnnxExample --self-test <example/assets>");
        return 2;
    }
    string config = selfTest ? Path.Combine(args[1], "test.json") : args[0];
    string imagePath = selfTest ? Path.Combine(args[1], "white.raw") : args[1];
    int width = selfTest ? 224 : int.Parse(args[2]);
    int height = selfTest ? 224 : int.Parse(args[3]);
    int channels = selfTest ? 1 : int.Parse(args[4]);
    int runs = !selfTest && args.Length == 6 ? int.Parse(args[5]) : 20;
    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3) || runs < 1 || runs > 10000)
        throw new ArgumentException("Positive width/height, channels 1 or 3 and runs 1..10000 required");
    byte[] pixels = File.ReadAllBytes(imagePath);
    if (pixels.Length != checked(width * height * channels))
        throw new ArgumentException("Raw size mismatch: use packed 8-bit GRAY/BGR, without a file header");

    // Load once. JSON selects ONNX, normalization, classes and the verified
    // optimizer/thread settings. Do not force ORT_ENABLE_ALL on the model.
    using var session = VisionSession.Open(config);
    for (int i = 0; i < 3; ++i) session.InferClassification(pixels, width, height, channels);
    double[] times = new double[runs];
    double pre = 0, model = 0, post = 0;
    ClassificationResult? result = null;
    for (int i = 0; i < runs; ++i)
    {
        long start = Stopwatch.GetTimestamp();
        result = session.InferClassification(pixels, width, height, channels);
        times[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        if (result.ClassId < 0 || !float.IsFinite(result.Confidence))
            throw new InvalidOperationException("Invalid inference result");
        if (selfTest && (result.ClassId != 0 || result.Probabilities.Length != 2 ||
            Math.Abs(result.Probabilities[0] - 0.88079708f) > 0.00001f ||
            Math.Abs(result.Probabilities[1] - 0.11920292f) > 0.00001f))
            throw new InvalidOperationException("Self-test output mismatch; check SDK/runtime settings");
        pre += result.PreprocessMilliseconds;
        model += result.ModelMilliseconds;
        post += result.PostprocessMilliseconds;
    }
    Array.Sort(times);
    Console.WriteLine($"class_id={result!.ClassId} class={result.ClassName} confidence={result.Confidence:F6}");
    Console.WriteLine($"runs={runs} preprocess_ms={pre / runs:F6} model_ms={model / runs:F6} postprocess_ms={post / runs:F6} call_p50_ms={times[(runs - 1) / 2]:F6} call_p95_ms={times[(int)Math.Ceiling(runs * .95) - 1]:F6}");
    if (selfTest) Console.WriteLine("PASS: ONNX load, inference and saved runtime settings");
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine($"FAIL: {error.Message}");
    return 1;
}
