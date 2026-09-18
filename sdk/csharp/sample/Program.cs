using DeepVisionStudio;

if (args.Length != 5)
{
    Console.Error.WriteLine("Usage: VisionRuntime.Sample.exe <bundle-dir> <raw-image> <width> <height> <channels>");
    Console.Error.WriteLine("The raw image must be packed 8-bit BGR/gray bytes.");
    return 2;
}

if (!int.TryParse(args[2], out var width) || !int.TryParse(args[3], out var height) ||
    !int.TryParse(args[4], out var channels) || width <= 0 || height <= 0 ||
    (channels != 1 && channels != 3))
{
    Console.Error.WriteLine("width, height and channels must describe a positive 1- or 3-channel image.");
    return 2;
}

var image = File.ReadAllBytes(args[1]);
var expected = checked(width * height * channels);
if (image.Length != expected)
{
    Console.Error.WriteLine($"Raw image size mismatch: expected {expected} bytes, got {image.Length}.");
    return 2;
}

using var session = VisionSession.OpenBundle(args[0], numThreads: 4);
var result = session.InferClassification(image, width, height, channels);
Console.WriteLine($"class_id={result.ClassId} class={result.ClassName} confidence={result.Confidence:R} total_ms={result.TotalMilliseconds:R}");
return 0;
