using System.Text;
using BertCppSharp;

class Program {
    public async static Task Main(string model, string text) {
        var gen = new BertGenerator();
        try {
            Console.WriteLine($"Loading model: {model}");
            await gen.LoadModelAsync(model);
        }
        catch (Exception e) {
            Console.WriteLine($"Failed to load model: {e}");
        }

        var wayTooLong = "This is a very long text that is way too long for the model to handle.";
        var sb = new StringBuilder();
        for (var i = 0; i < 100; i++) {
            sb.Append(wayTooLong);
        }
        text = sb.ToString();
        
        Console.WriteLine($"Model: {gen.ModelFile}");
        Console.WriteLine($"Max sequence length: {gen.MaxSequenceLength}");
        Console.WriteLine($"Embedding size: {gen.EmbeddingSize}");
        
        Console.WriteLine($"Evaluate: {text}");
        // embed a single text
        var embedding = gen.Embed(text);
        
        Console.WriteLine($"Embedding: {string.Join(", ", embedding)}");
    }
}