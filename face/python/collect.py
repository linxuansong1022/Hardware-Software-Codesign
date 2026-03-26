import argparse
import csv
import os
import sys
import time
from datetime import datetime

import pygame
import serial

# Configuration
DEFAULT_OUTPUT_PATH = "../data/"
DEFAULT_PORT = "COM3"
BAUD_RATE = 921600
SERIAL_TIMEOUT = 2.0
WIDTH = 320
HEIGHT = 240
FRAME_PREAMBLE = b"===FRAME===\n"

# Key → folder name mapping
CLASS_MAP: dict[int, str] = {
    pygame.K_0: "person_a",
    pygame.K_1: "person_b",
    pygame.K_2: "person_c",
    pygame.K_3: "unknown",
}

CLASS_LABELS: dict[int, str] = {
    pygame.K_0: "0: person_a",
    pygame.K_1: "1: person_b",
    pygame.K_2: "2: person_c",
    pygame.K_3: "3: unknown",
}


def capture_and_display_loop(port: str, output_path: str) -> None:
    print(f"Opening serial port {port}... ", end="")
    try:
        serial_port = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        serial_port.reset_input_buffer()
    except serial.SerialException as exc:
        print(f"Failed to open serial port {port}: {exc}", file=sys.stderr)
        return

    pygame.init()
    font = pygame.font.SysFont("monospace", 16)
    screen = pygame.display.set_mode((WIDTH, HEIGHT + 60))
    pygame.display.set_caption("Face Data Collector")

    print("Connection established.")
    print("Keys: 0=person_a  1=person_b  2=person_c  3=unknown  q=quit")

    serial_port.write(b"S")

    # Count saved images per class
    counts: dict[str, int] = {name: _count_existing(output_path, name) for name in CLASS_MAP.values()}

    last_surface: pygame.Surface | None = None
    last_saved_label: str = ""
    last_saved_time: float = 0.0
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key in CLASS_MAP and last_surface is not None:
                        class_name = CLASS_MAP[event.key]
                        _save_frame(output_path, last_surface, class_name)
                        counts[class_name] += 1
                        last_saved_label = f"Saved → {class_name}"
                        last_saved_time = time.time()

            surface = _capture_frame(serial_port)
            if surface is None:
                continue
            last_surface = surface.copy()

            # Draw camera feed
            screen.fill((30, 30, 30))
            screen.blit(surface, (0, 0))

            # Draw HUD bar below the image
            hud_y = HEIGHT + 5
            count_text = "  ".join(
                f"{CLASS_LABELS[k]}({counts[CLASS_MAP[k]]})"
                for k in sorted(CLASS_MAP)
            )
            screen.blit(font.render(count_text, True, (200, 200, 200)), (5, hud_y))

            # Show "Saved" flash for 1.5 seconds
            if last_saved_label and time.time() - last_saved_time < 1.5:
                screen.blit(font.render(last_saved_label, True, (100, 255, 100)), (5, hud_y + 20))

            pygame.display.flip()
            time.sleep(0.001)
    finally:
        serial_port.close()
        pygame.quit()


def _capture_frame(serial_port: serial.Serial) -> pygame.Surface | None:
    chunk = serial_port.read_until(FRAME_PREAMBLE)
    if not chunk.endswith(FRAME_PREAMBLE):
        print("Preamble timeout, retrying...")
        return None

    frame_rgb565 = serial_port.read(WIDTH * HEIGHT * 2)
    if len(frame_rgb565) != WIDTH * HEIGHT * 2:
        print(f"Incomplete frame ({len(frame_rgb565)} bytes), skipping...")
        return None

    frame_rgb = bytearray(WIDTH * HEIGHT * 3)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            src = (y * WIDTH + x) * 2
            dst = (y * WIDTH + x) * 3
            b1 = frame_rgb565[src]
            b2 = frame_rgb565[src + 1]
            frame_rgb[dst]     = b1 & 0xF8
            frame_rgb[dst + 1] = ((b1 & 0x07) << 5) | ((b2 & 0xE0) >> 3)
            frame_rgb[dst + 2] = (b2 & 0x1F) << 3

    return pygame.image.frombuffer(bytes(frame_rgb), (WIDTH, HEIGHT), "RGB")


def _save_frame(output_path: str, surface: pygame.Surface, class_name: str) -> None:
    directory = os.path.join(output_path, class_name)
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(directory, f"image_{timestamp}.png")
    pygame.image.save(surface, path)
    print(f"Saved {path}")


def _count_existing(output_path: str, class_name: str) -> int:
    directory = os.path.join(output_path, class_name)
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face data collector")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Output directory")
    args = parser.parse_args()
    capture_and_display_loop(args.port, args.output_path)
