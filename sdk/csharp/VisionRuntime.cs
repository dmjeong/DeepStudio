using System.Runtime.InteropServices;

namespace DeepVisionStudio;

public enum VisionResultKind : uint
{
    Classification = 1,
    Segmentation = 2,
    Detection = 3,
    Anomaly = 4,
}

public sealed record ClassificationResult(
    int ClassId,
    string ClassName,
    float Confidence,
    float[] Probabilities,
    double TotalMilliseconds,
    double PreprocessMilliseconds,
    double ModelMilliseconds,
    double PostprocessMilliseconds);

public sealed record SegmentationResult(
    int Width,
    int Height,
    int Classes,
    byte[] Mask,
    double TotalMilliseconds);

public sealed record Detection(
    float X1,
    float Y1,
    float X2,
    float Y2,
    int ClassId,
    float Confidence);

public sealed record DetectionResult(
    Detection[] Detections,
    double TotalMilliseconds,
    double PreprocessMilliseconds,
    double ModelMilliseconds,
    double PostprocessMilliseconds);

public sealed record AnomalyResult(
    int Width,
    int Height,
    float[] Map,
    float Score,
    float Threshold,
    bool IsAnomalous,
    double TotalMilliseconds,
    double PreprocessMilliseconds,
    double ModelMilliseconds,
    double PostprocessMilliseconds);

public sealed record SamPrompt(
    float[] PointCoordinates,
    int[] PointLabels,
    float[]? BoxXYXY = null,
    float[]? MaskInput = null,
    int MaskWidth = 256,
    int MaskHeight = 256);

public sealed class VisionSession : SafeHandle
{
    private const string NativeLibrary = "vision_runtime";
    private const uint AbiVersion = 1;

    private VisionSession(IntPtr handle) : base(IntPtr.Zero, true) => SetHandle(handle);

    public static VisionSession Open(string configPath, string runtime = "onnxruntime", int numThreads = -1)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(configPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(runtime);
        if (numThreads < -1) throw new ArgumentOutOfRangeException(nameof(numThreads));

        IntPtr runtimeUtf8 = Marshal.StringToCoTaskMemUTF8(runtime);
        try
        {
            var options = new NativeSessionOptions
            {
                StructSize = (uint)Marshal.SizeOf<NativeSessionOptions>(),
                AbiVersion = AbiVersion,
                RuntimeUtf8 = runtimeUtf8,
                NumThreads = numThreads,
            };
            var status = Native.dv_create_session(configPath, ref options, out var raw);
            if (status != 0 || raw == IntPtr.Zero)
                throw new InvalidOperationException($"Vision session creation failed ({StatusName(status)}).");
            return new VisionSession(raw);
        }
        finally { Marshal.FreeCoTaskMem(runtimeUtf8); }
    }

    public static VisionSession OpenBundle(string bundlePath, string runtime = "onnxruntime", int numThreads = -1)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(bundlePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(runtime);
        if (numThreads < -1) throw new ArgumentOutOfRangeException(nameof(numThreads));

        IntPtr runtimeUtf8 = Marshal.StringToCoTaskMemUTF8(runtime);
        try
        {
            var options = new NativeSessionOptions
            {
                StructSize = (uint)Marshal.SizeOf<NativeSessionOptions>(),
                AbiVersion = AbiVersion,
                RuntimeUtf8 = runtimeUtf8,
                NumThreads = numThreads,
            };
            var status = Native.dv_create_session_from_bundle(bundlePath, ref options, out var raw);
            if (status != 0 || raw == IntPtr.Zero)
                throw new InvalidOperationException($"Vision bundle session creation failed ({StatusName(status)}).");
            return new VisionSession(raw);
        }
        finally { Marshal.FreeCoTaskMem(runtimeUtf8); }
    }

    public ClassificationResult InferClassification(byte[] image, int width, int height, int channels,
                                                     int strideBytes = 0)
    {
        var result = Infer(image, width, height, channels, strideBytes);
        if (result.Kind != VisionResultKind.Classification)
            throw new InvalidOperationException("The loaded model is not a classification model.");
        return result.Classification!;
    }

    public SegmentationResult InferSegmentation(byte[] image, int width, int height, int channels,
                                                 int strideBytes = 0)
    {
        var result = Infer(image, width, height, channels, strideBytes);
        if (result.Kind != VisionResultKind.Segmentation)
            throw new InvalidOperationException("The loaded model is not a segmentation model.");
        return result.Segmentation!;
    }

    public DetectionResult InferDetection(byte[] image, int width, int height, int channels,
                                          int strideBytes = 0)
    {
        var result = Infer(image, width, height, channels, strideBytes);
        if (result.Kind != VisionResultKind.Detection)
            throw new InvalidOperationException("The loaded model is not a detection model.");
        return result.Detection!;
    }

    public AnomalyResult InferAnomaly(byte[] image, int width, int height, int channels,
                                      int strideBytes = 0)
    {
        var result = Infer(image, width, height, channels, strideBytes);
        if (result.Kind != VisionResultKind.Anomaly)
            throw new InvalidOperationException("The loaded model is not an anomaly model.");
        return result.Anomaly!;
    }

    public SamImageContext EncodeSam(byte[] image, int width, int height, int channels,
                                     int strideBytes = 0)
    {
        ObjectDisposedException.ThrowIf(IsInvalid, this);
        ArgumentNullException.ThrowIfNull(image);
        var view = PinImage(image, width, height, channels, strideBytes, out var pin);
        try
        {
            var status = Native.dv_sam_encode(handle, ref view, out var raw);
            if (status != 0 || raw == IntPtr.Zero)
                throw new InvalidOperationException($"SAM2 encode failed ({StatusName(status)}): {LastError}");
            return new SamImageContext(this, raw);
        }
        finally { pin.Free(); }
    }

    public SegmentationResult SegmentSam(SamImageContext context, SamPrompt prompt)
    {
        ObjectDisposedException.ThrowIf(IsInvalid, this);
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(prompt);
        if (context.Owner != this || context.IsInvalid)
            throw new ArgumentException("The SAM2 image context belongs to another session.", nameof(context));
        ArgumentNullException.ThrowIfNull(prompt.PointCoordinates);
        ArgumentNullException.ThrowIfNull(prompt.PointLabels);
        if ((prompt.PointCoordinates.Length & 1) != 0 ||
            prompt.PointCoordinates.Length / 2 != prompt.PointLabels.Length)
            throw new ArgumentException("SAM2 point coordinates and labels must have matching lengths.", nameof(prompt));
        if (prompt.BoxXYXY is not null && prompt.BoxXYXY.Length != 4)
            throw new ArgumentException("SAM2 box prompt must contain four values.", nameof(prompt));
        if (prompt.MaskInput is not null &&
            (prompt.MaskWidth <= 0 || prompt.MaskHeight <= 0 ||
             prompt.MaskInput.Length != checked(prompt.MaskWidth * prompt.MaskHeight)))
            throw new ArgumentException("SAM2 mask prompt dimensions do not match the buffer.", nameof(prompt));

        GCHandle points = default, labels = default, box = default, mask = default;
        try
        {
            var pointPtr = PinIfNonEmpty(prompt.PointCoordinates, ref points);
            var labelPtr = PinIfNonEmpty(prompt.PointLabels, ref labels);
            var boxPtr = prompt.BoxXYXY is null ? IntPtr.Zero : PinIfNonEmpty(prompt.BoxXYXY, ref box);
            var maskPtr = prompt.MaskInput is null ? IntPtr.Zero : PinIfNonEmpty(prompt.MaskInput, ref mask);
            var nativePrompt = new NativeSamPrompt
            {
                StructSize = (uint)Marshal.SizeOf<NativeSamPrompt>(),
                AbiVersion = AbiVersion,
                PointCoordinatesXY = pointPtr,
                PointLabels = labelPtr,
                PointCount = (uint)prompt.PointLabels.Length,
                BoxXYXY = boxPtr,
                MaskInput = maskPtr,
                MaskWidth = prompt.MaskInput is null ? 0u : (uint)prompt.MaskWidth,
                MaskHeight = prompt.MaskInput is null ? 0u : (uint)prompt.MaskHeight,
            };
            var status = Native.dv_sam_segment(handle, context.DangerousGetHandle(), ref nativePrompt, out var raw);
            if (status != 0 || raw == IntPtr.Zero)
                throw new InvalidOperationException($"SAM2 prompt failed ({StatusName(status)}): {LastError}");
            try
            {
                var result = NativeResultCopy.From(raw);
                if (result.Kind != VisionResultKind.Segmentation)
                    throw new InvalidOperationException("SAM2 returned a non-segmentation result.");
                return result.Segmentation!;
            }
            finally { Native.dv_release_result(raw); }
        }
        finally
        {
            if (points.IsAllocated) points.Free();
            if (labels.IsAllocated) labels.Free();
            if (box.IsAllocated) box.Free();
            if (mask.IsAllocated) mask.Free();
        }
    }

    private static IntPtr PinIfNonEmpty(Array value, ref GCHandle handle)
    {
        if (value.Length == 0) return IntPtr.Zero;
        handle = GCHandle.Alloc(value, GCHandleType.Pinned);
        return handle.AddrOfPinnedObject();
    }

    private static NativeImageView PinImage(byte[] image, int width, int height, int channels,
                                            int strideBytes, out GCHandle pin)
    {
        if (width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4))
            throw new ArgumentOutOfRangeException(nameof(width));
        var packed = checked(width * channels);
        var stride = strideBytes == 0 ? packed : strideBytes;
        if (stride < packed || image.Length < checked(stride * height))
            throw new ArgumentException("Image buffer is smaller than the declared view.", nameof(image));
        pin = GCHandle.Alloc(image, GCHandleType.Pinned);
        return new NativeImageView
        {
            StructSize = (uint)Marshal.SizeOf<NativeImageView>(),
            AbiVersion = AbiVersion,
            Data = pin.AddrOfPinnedObject(),
            Width = width,
            Height = height,
            Channels = channels,
            StrideBytes = stride,
        };
    }

    private NativeResultCopy Infer(byte[] image, int width, int height, int channels, int strideBytes)
    {
        ObjectDisposedException.ThrowIf(IsInvalid, this);
        ArgumentNullException.ThrowIfNull(image);
        if (width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4))
            throw new ArgumentOutOfRangeException(nameof(width));
        var packed = checked(width * channels);
        var stride = strideBytes == 0 ? packed : strideBytes;
        if (stride < packed || image.Length < checked(stride * height))
            throw new ArgumentException("Image buffer is smaller than the declared view.", nameof(image));

        var pin = GCHandle.Alloc(image, GCHandleType.Pinned);
        try
        {
            var view = new NativeImageView
            {
                StructSize = (uint)Marshal.SizeOf<NativeImageView>(),
                AbiVersion = AbiVersion,
                Data = pin.AddrOfPinnedObject(),
                Width = width,
                Height = height,
                Channels = channels,
                StrideBytes = stride,
            };
            var status = Native.dv_infer(handle, ref view, out var raw);
            if (status != 0 || raw == IntPtr.Zero)
                throw new InvalidOperationException($"Vision inference failed ({StatusName(status)}): {LastError}");
            try { return NativeResultCopy.From(raw); }
            finally { Native.dv_release_result(raw); }
        }
        finally { pin.Free(); }
    }

    public string LastError => Marshal.PtrToStringUTF8(Native.dv_last_error(handle)) ?? string.Empty;

    private static string StatusName(int status) =>
        Marshal.PtrToStringUTF8(Native.dv_status_name(status)) ?? "unknown";

    protected override bool ReleaseHandle()
    {
        Native.dv_close_session(handle);
        return true;
    }

    private sealed record NativeResultCopy(VisionResultKind Kind, ClassificationResult? Classification,
                                           SegmentationResult? Segmentation, DetectionResult? Detection,
                                           AnomalyResult? Anomaly)
    {
        public static NativeResultCopy From(IntPtr pointer)
        {
            var native = Marshal.PtrToStructure<NativeResult>(pointer);
            if (native.Kind == VisionResultKind.Classification)
            {
                var probabilities = new float[checked((int)native.ProbabilityCount)];
                if (probabilities.Length != 0) Marshal.Copy(native.Probabilities, probabilities, 0, probabilities.Length);
                return new NativeResultCopy(native.Kind,
                    new ClassificationResult(native.ClassId,
                        Marshal.PtrToStringUTF8(native.ClassNameUtf8) ?? string.Empty,
                        native.Confidence, probabilities, native.TotalMs, native.PreprocessMs,
                        native.ModelMs, native.PostprocessMs), null, null, null);
            }
            if (native.Kind == VisionResultKind.Segmentation)
            {
                var mask = new byte[checked((int)(native.MaskWidth * native.MaskHeight))];
                for (uint row = 0; row < native.MaskHeight; row++)
                    Marshal.Copy(IntPtr.Add(native.Mask, checked((int)(row * native.MaskStrideBytes))),
                        mask, checked((int)(row * native.MaskWidth)), checked((int)native.MaskWidth));
                return new NativeResultCopy(native.Kind, null,
                    new SegmentationResult((int)native.MaskWidth, (int)native.MaskHeight,
                        (int)native.MaskClasses, mask, native.TotalMs), null, null);
            }
            if (native.Kind == VisionResultKind.Detection)
            {
                var detections = new Detection[checked((int)native.DetectionCount)];
                var size = Marshal.SizeOf<NativeDetection>();
                for (var index = 0; index < detections.Length; index++)
                {
                    var item = Marshal.PtrToStructure<NativeDetection>(
                        IntPtr.Add(native.Detections, checked(index * size)));
                    detections[index] = new Detection(item.X1, item.Y1, item.X2, item.Y2,
                        item.ClassId, item.Confidence);
                }
                return new NativeResultCopy(native.Kind, null, null,
                    new DetectionResult(detections, native.TotalMs, native.PreprocessMs,
                        native.ModelMs, native.PostprocessMs), null);
            }
            if (native.Kind == VisionResultKind.Anomaly)
            {
                var values = checked((int)(native.AnomalyMapWidth * native.AnomalyMapHeight));
                var map = new float[values];
                for (uint row = 0; row < native.AnomalyMapHeight; row++)
                    Marshal.Copy(IntPtr.Add(native.AnomalyMap,
                                            checked((int)(row * native.AnomalyMapStrideBytes))),
                        map, checked((int)(row * native.AnomalyMapWidth)), checked((int)native.AnomalyMapWidth));
                return new NativeResultCopy(native.Kind, null, null, null,
                    new AnomalyResult((int)native.AnomalyMapWidth, (int)native.AnomalyMapHeight,
                        map, native.AnomalyScore, native.AnomalyThreshold, native.Anomalous != 0,
                        native.TotalMs, native.PreprocessMs, native.ModelMs, native.PostprocessMs));
            }
            throw new InvalidOperationException("Unknown native result kind.");
        }
    }

    public sealed class SamImageContext : SafeHandle
    {
        internal SamImageContext(VisionSession owner, IntPtr handle) : base(IntPtr.Zero, true)
        {
            Owner = owner;
            SetHandle(handle);
        }

        internal VisionSession Owner { get; }

        protected override bool ReleaseHandle()
        {
            Native.dv_release_image_context(handle);
            return true;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeSessionOptions
    {
        public uint StructSize;
        public uint AbiVersion;
        public IntPtr RuntimeUtf8;
        public int NumThreads;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeImageView
    {
        public uint StructSize;
        public uint AbiVersion;
        public IntPtr Data;
        public int Width;
        public int Height;
        public int Channels;
        public int StrideBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeSamPrompt
    {
        public uint StructSize;
        public uint AbiVersion;
        public IntPtr PointCoordinatesXY;
        public IntPtr PointLabels;
        public uint PointCount;
        public IntPtr BoxXYXY;
        public IntPtr MaskInput;
        public uint MaskWidth;
        public uint MaskHeight;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeResult
    {
        public uint StructSize;
        public uint AbiVersion;
        public VisionResultKind Kind;
        public int ClassId;
        public float Confidence;
        public IntPtr Probabilities;
        public uint ProbabilityCount;
        public IntPtr ClassNameUtf8;
        public IntPtr Mask;
        public uint MaskWidth;
        public uint MaskHeight;
        public uint MaskStrideBytes;
        public uint MaskClasses;
        public double TotalMs;
        public double PreprocessMs;
        public double ModelMs;
        public double PostprocessMs;
        public IntPtr Detections;
        public uint DetectionCount;
        public float AnomalyScore;
        public float AnomalyThreshold;
        public uint Anomalous;
        public IntPtr AnomalyMap;
        public uint AnomalyMapWidth;
        public uint AnomalyMapHeight;
        public uint AnomalyMapStrideBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeDetection
    {
        public float X1;
        public float Y1;
        public float X2;
        public float Y2;
        public int ClassId;
        public float Confidence;
    }

    private static class Native
    {
        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int dv_create_session([MarshalAs(UnmanagedType.LPUTF8Str)] string configPath,
            ref NativeSessionOptions options, out IntPtr session);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int dv_create_session_from_bundle([MarshalAs(UnmanagedType.LPUTF8Str)] string bundlePath,
            ref NativeSessionOptions options, out IntPtr session);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int dv_infer(IntPtr session, ref NativeImageView image, out IntPtr result);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int dv_sam_encode(IntPtr session, ref NativeImageView image, out IntPtr context);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int dv_sam_segment(IntPtr session, IntPtr context,
            ref NativeSamPrompt prompt, out IntPtr result);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr dv_last_error(IntPtr session);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr dv_status_name(int status);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void dv_release_result(IntPtr result);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void dv_release_image_context(IntPtr context);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void dv_close_session(IntPtr session);
    }
}
