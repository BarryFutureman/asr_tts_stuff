from pytube import YouTube

# Function to download video
def download_video(url, resolution='720p'):
    yt = YouTube(url)

    streams = yt.streams.filter(adaptive=True, file_extension='mp4')
    resolutions = set()
    for stream in streams:
        resolutions.add(stream.resolution)
    print("Available Resolutions:")
    print(resolutions)
    quit()

    if stream:
        stream.download()
        print("Download completed successfully.")
    else:
        print("No stream found for the specified resolution.")

# Example usage
url = 'https://www.youtube.com/watch?v=iL5QYU0n5n8'
download_video(url, resolution='1080p60')  # Change resolution as needed