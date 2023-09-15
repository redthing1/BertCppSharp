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
        
        
        Console.WriteLine($"Model: {gen.ModelFile}");
        Console.WriteLine($"Max sequence length: {gen.MaxSequenceLength}");
        Console.WriteLine($"Embedding size: {gen.EmbeddingSize}");
        
        Console.WriteLine($"Evaluate: {text}");
        // embed a single text
        var embedding = gen.Embed(text);
        
        Console.WriteLine($"Embedding: {string.Join(", ", embedding)}");
    }
}