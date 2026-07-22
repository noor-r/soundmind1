using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Avalonia;
using Avalonia.WebView.Desktop;
using YoutubeDownloader.Core.Downloading;
using YoutubeDownloader.Utils;
using YoutubeExplode;
using YoutubeExplode.Videos;
using YoutubeExplode.Videos.Streams;

namespace YoutubeDownloader;

public static class Program
{
    private static Assembly Assembly { get; } = Assembly.GetExecutingAssembly();

    public static string Name { get; } = Assembly.GetName().Name ?? "YoutubeDownloader";

    public static Version Version { get; } = Assembly.GetName().Version ?? new Version(0, 0, 0);

    public static string VersionString { get; } = Version.ToString(3);

    public static bool IsDevelopmentBuild { get; } = Version.Major is <= 0 or >= 999;

    public static string ProjectUrl { get; } = "https://github.com/Tyrrrz/YoutubeDownloader";

    public static string ProjectReleasesUrl { get; } = $"{ProjectUrl}/releases";

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>().UsePlatformDetect().LogToTrace().UseDesktopWebView();

    [STAThread]
    public static int Main(string[] args)
    {
        if (TryHandleCliDownload(args, out var exitCode))
            return exitCode;

        // Build and run the app
        var builder = BuildAvaloniaApp();

        try
        {
            return builder.StartWithClassicDesktopLifetime(args);
        }
        catch (Exception ex)
        {
            if (OperatingSystem.IsWindows())
                _ = NativeMethods.Windows.MessageBox(0, ex.ToString(), "Fatal Error", 0x10);

            throw;
        }
        finally
        {
            // Clean up after application shutdown
            if (builder.Instance is IDisposable disposableApp)
                disposableApp.Dispose();
        }
    }

    private static bool TryHandleCliDownload(string[] args, out int exitCode)
    {
        exitCode = 0;

        if (args.Length == 0)
            return false;

        var firstArg = args[0];
        if (firstArg is "-h" or "--help")
        {
            Console.WriteLine(
                "Usage: dotnet run --project YoutubeDownloader/YoutubeDownloader.csproj -- <youtube-url> [output-file.mp4]"
            );
            exitCode = 0;
            return true;
        }

        if (args.Length > 2)
        {
            Console.Error.WriteLine(
                "Expected at most two arguments: <youtube-url> [output-file.mp4]"
            );
            exitCode = 1;
            return true;
        }

        var url = firstArg;
        var outputFilePath =
            args.Length > 1
                ? args[1]
                : Path.Combine(
                    Directory.GetCurrentDirectory(),
                    $"{ExtractVideoId(url) ?? "video"}.mp4"
                );

        if (string.IsNullOrWhiteSpace(url))
        {
            Console.Error.WriteLine("A YouTube URL is required.");
            exitCode = 1;
            return true;
        }

        var videoId = ExtractVideoId(url);
        if (videoId is null)
        {
            Console.Error.WriteLine($"Unable to parse YouTube video URL: {url}");
            exitCode = 1;
            return true;
        }

        try
        {
            var task = DownloadLowestQualityVideoAsync(videoId, outputFilePath);
            task.GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            exitCode = 1;
            return true;
        }

        Console.WriteLine($"Downloaded to {Path.GetFullPath(outputFilePath)}");
        return true;
    }

    private static async System.Threading.Tasks.Task DownloadLowestQualityVideoAsync(
        string videoId,
        string outputFilePath
    )
    {
        if (FFmpeg.TryGetCliFilePath() is null)
        {
            Console.WriteLine("FFmpeg was not found. Downloading it now...");
            await FFmpeg.DownloadAsync(Path.Combine(AppContext.BaseDirectory, FFmpeg.CliFileName));
        }

        var youtube = new YoutubeClient();
        var video = await youtube.Videos.GetAsync(videoId);

        var outputPath = Path.GetFullPath(outputFilePath);
        var outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        using var downloader = new VideoDownloader();
        var downloadOption = await downloader.GetBestDownloadOptionAsync(
            video.Id,
            new VideoDownloadPreference(Container.Mp4, VideoQualityPreference.Lowest)
        );

        await downloader.DownloadVideoAsync(
            outputPath,
            video,
            downloadOption,
            includeSubtitles: false,
            ffmpegPath: FFmpeg.TryGetCliFilePath() ?? "ffmpeg"
        );
    }

    private static string? ExtractVideoId(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
            return null;

        if (VideoId.TryParse(input) is { } directVideoId)
            return directVideoId;

        if (!Uri.TryCreate(input, UriKind.Absolute, out var uri))
            return null;

        if (
            uri.Host.Contains("youtube.com", StringComparison.OrdinalIgnoreCase)
            || uri.Host.Contains("youtu.be", StringComparison.OrdinalIgnoreCase)
        )
        {
            if (uri.Query is { Length: > 0 })
            {
                foreach (
                    var pair in uri
                        .Query.TrimStart('?')
                        .Split('&', StringSplitOptions.RemoveEmptyEntries)
                )
                {
                    var parts = pair.Split('=', 2);
                    if (
                        parts[0].Equals("v", StringComparison.OrdinalIgnoreCase)
                        && parts.Length > 1
                        && VideoId.TryParse(parts[1]) is { } queryVideoId
                    )
                        return queryVideoId;
                }
            }

            var segments = uri.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries);
            if (segments.Length > 0)
            {
                var lastSegment = segments.Last();
                if (
                    lastSegment.Length is > 0 and < 30
                    && !lastSegment.Equals("watch", StringComparison.OrdinalIgnoreCase)
                    && VideoId.TryParse(lastSegment) is { } pathVideoId
                )
                    return pathVideoId;
            }
        }

        return null;
    }
}
