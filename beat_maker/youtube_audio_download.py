# === IMPORTANT ===
# pip install pytube
# ^^ run this in terminal before you run this script

from pytube import YouTube


def download_audio_video(url):
    # Create a YouTube object
    youtube = YouTube(url)

    # Get the highest resolution audio stream
    audio_stream = youtube.streams.filter(only_audio=True).first()

    # Download the audio stream
    audio_stream.download()

    # Return the filename of the downloaded audio
    return audio_stream.default_filename


# ===================================================================
if __name__ == '__main__':
    # Put the YouTube Video link here
    youtube_video_link = ""

    downloaded_filename = download_audio_video(youtube_video_link)

    print("File saved as:", downloaded_filename)
