import librosa
import pygame
import numpy as np

# Function to visualize audio using pygame
def visualize_audio(filename):
    # Load audio file
    y, sr = librosa.load(filename)

    # Compute amplitude envelope
    amplitude_envelope = np.abs(librosa.stft(y))

    # Normalize amplitude envelope
    amplitude_envelope /= np.max(amplitude_envelope)

    # Set up pygame
    pygame.init()
    clock = pygame.time.Clock()

    # Set up display
    display_width = 800
    display_height = 400
    game_display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('Audio Visualization')
    game_display.fill((255, 255, 255))

    # Main loop for visualization
    running = True
    t = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game_display.fill((255, 255, 255))

        # Draw amplitude envelope
        for i in range(display_width):
            print(amplitude_envelope)
            quit()
            end_h = display_height - amplitude_envelope[int((i/display_width)*len(amplitude_envelope))]*display_height
            print(end_h)
            quit()
            pygame.draw.line(game_display, (0, 0, 0), (i, display_height), (i, end_h), 1)

        pygame.display.update()
        clock.tick(60)

    pygame.quit()

# Example usage
if __name__ == "__main__":
    audio_file = 'songs/NEONI - VILLAIN (Lyrics).mp3'  # Change this to your audio file path
    visualize_audio(audio_file)
