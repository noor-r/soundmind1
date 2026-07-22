using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using YoutubeDownloader.Core.Downloading;
using YoutubeDownloader.Core.Resolving;
using YoutubeExplode;
using YoutubeExplode.Common;
using YoutubeExplode.Videos;
using YoutubeExplode.Videos.Streams;

if (args.Length == 2 && args[0].Equals("search", StringComparison.OrdinalIgnoreCase))
{
    try
    {
        using var resolver = new QueryResolver();
        var search = await resolver.ResolveAsync($"?{args[1]}");
        var results = search.Videos.Take(5).Select(video => new
        {
            title = video.Title,
            author = video.Author.ChannelTitle,
            url = video.Url,
            durationSeconds = video.Duration?.TotalSeconds,
            thumbnail = video.Thumbnails.TryGetWithHighestResolution()?.Url,
        });

        Console.WriteLine(JsonSerializer.Serialize(results));
        return 0;
    }
    catch (Exception exception)
    {
        Console.Error.WriteLine(exception.Message);
        return 1;
    }
}

if (args.Length != 2)
{
    Console.Error.WriteLine(
        "Usage: SoundMind.Downloader <youtube-url> <output-file.mp4> | search <song-name>"
    );
    return 1;
}

if (VideoId.TryParse(args[0]) is not { } videoId)
{
    Console.Error.WriteLine("The supplied value is not a valid YouTube video URL or ID.");
    return 1;
}

var outputPath = Path.GetFullPath(args[1]);
if (!outputPath.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase))
{
    Console.Error.WriteLine("The output file must use the .mp4 extension.");
    return 1;
}

try
{
    var ffmpegPath = FFmpeg.TryGetCliFilePath();
    if (ffmpegPath is null)
    {
        ffmpegPath = Path.Combine(AppContext.BaseDirectory, FFmpeg.CliFileName);
        Console.WriteLine("FFmpeg was not found; downloading it for the local downloader...");
        await FFmpeg.DownloadAsync(ffmpegPath);
    }

    var outputDirectory = Path.GetDirectoryName(outputPath);
    if (!string.IsNullOrWhiteSpace(outputDirectory))
        Directory.CreateDirectory(outputDirectory);

    var youtube = new YoutubeClient();
    var video = await youtube.Videos.GetAsync(videoId);

    using var downloader = new VideoDownloader();
    var option = await downloader.GetBestDownloadOptionAsync(
        video.Id,
        new VideoDownloadPreference(Container.Mp4, VideoQualityPreference.Lowest)
    );

    await downloader.DownloadVideoAsync(
        outputPath,
        video,
        option,
        includeSubtitles: false,
        ffmpegPath: ffmpegPath
    );

    Console.WriteLine($"Downloaded lowest-quality MP4 to {outputPath}");
    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine(exception.Message);
    return 1;
}