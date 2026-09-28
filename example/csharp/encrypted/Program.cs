using DeepVisionStudio;
using System.Security.Cryptography;

// No arguments. Keep session as a member in your application, initialized once.
string root = AppContext.BaseDirectory;
byte[] key = File.ReadAllBytes(Path.Combine(root, "assets", "example-only.key"));
VisionSession session;
try { session = VisionSession.OpenEncrypted(Path.Combine(root, "assets", "test.dvsenc"), key); }
finally { CryptographicOperations.ZeroMemory(key); }
using (session)
{
    byte[] image = File.ReadAllBytes(Path.Combine(root, "assets", "white.raw"));
    // Set this to your exported task: classify, detect, segment, anomaly, sam2.
    const string task = "classify";
    void InferFrame()
    {
        switch (task)
        {
            case "classify":
                var c = session.InferClassification(image, 224, 224, 1);
                Console.WriteLine($"{c.ClassName} confidence={c.Confidence} ms={c.TotalMilliseconds}");
                break;
            case "detect":
                Console.WriteLine(session.InferDetection(image, 224, 224, 1)); break;
            case "segment":
                Console.WriteLine(session.InferSegmentation(image, 224, 224, 1)); break;
            case "anomaly":
                Console.WriteLine(session.InferAnomaly(image, 224, 224, 1)); break;
            case "sam2":
                using (var context = session.EncodeSam(image, 224, 224, 1))
                    Console.WriteLine(session.AutomaticSam(context, 2, 2));
                break;
            default: throw new InvalidOperationException("Unknown task");
        }
    }
    InferFrame(); // Startup warm-up, before accepting inspection requests.
    for (int i = 0; i < 20; ++i) InferFrame();
}
