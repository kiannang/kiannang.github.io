import pygame
import numpy as np
import sounddevice as sd
from numpy.fft import rfft

# ================= CONFIG =================
WIDTH, HEIGHT = 900, 600
DEVICE_INDEX = 18
FPS = 60

# Color (high-contrast ice)
ICE_BLUE = np.array([110, 175, 255])   # darker, richer blue
PURE_WHITE = np.array([255, 255, 255]) # true white
COLOR_SMOOTH = 0.06


# Audio
SMOOTHING = 0.65
BASS_GAIN = 1.6

# Snowflake geometry
ARMS = 6
ARM_LENGTH = 260
ARM_THICKNESS = 7

INNER_RING_RADIUS = 120
RING_THICKNESS = 7

BRANCH_ANGLE = np.pi / 6
BRANCH_LENGTH = 70
BRANCH_THICKNESS = 5

SUB_BRANCH_ANGLE = np.pi / 10
SUB_BRANCH_LENGTH = 45
SUB_BRANCH_THICKNESS = 4

# Visual style
PIXEL_SIZE = 2
FUZZ_RADIUS = 6

# Motion
FLOW_STRENGTH = 0.08

# 🔑 SCALE (unchanged behavior)
BASE_SCALE = 0.28
BASS_SCALE = 2
SCALE_SPRING = 0.25
SCALE_DAMPING = 0.75

# ✅ NEW: GLOBAL GEOMETRY SCALE
GLOBAL_SCALE = 0.05   # <<< makes whole snowflake smaller
# =========================================

# ---------- AUDIO ----------
bass = 0.0
bass_s = 0.0

def audio_callback(indata, frames, time, status):
    global bass
    samples = indata[:, 0]
    spectrum = np.abs(rfft(samples))
    bass = np.mean(spectrum[2:12])

stream = sd.InputStream(
    device=DEVICE_INDEX,
    channels=2,
    samplerate=48000,
    callback=audio_callback
)
stream.start()

# ---------- BUILD SNOWFLAKE MASK ----------
cx, cy = WIDTH // 2, HEIGHT // 2
mask = []

def draw_thick_line(x0, y0, x1, y1, thickness):
    steps = int(np.hypot(x1 - x0, y1 - y0))
    for t in np.linspace(0, 1, steps):
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                mask.append((int(x + dx), int(y + dy)))

sector = 2 * np.pi / ARMS

current_color = PURE_WHITE.astype(float)

for i in range(ARMS):
    a = i * sector
    draw_thick_line(cx, cy,
                    cx + np.cos(a) * ARM_LENGTH,
                    cy + np.sin(a) * ARM_LENGTH,
                    ARM_THICKNESS)

    for s in (-1, 1):
        draw_thick_line(
            cx + np.cos(a) * ARM_LENGTH * 0.55,
            cy + np.sin(a) * ARM_LENGTH * 0.55,
            cx + np.cos(a + s * BRANCH_ANGLE) * (ARM_LENGTH * 0.55 + BRANCH_LENGTH),
            cy + np.sin(a + s * BRANCH_ANGLE) * (ARM_LENGTH * 0.55 + BRANCH_LENGTH),
            BRANCH_THICKNESS
        )

        draw_thick_line(
            cx + np.cos(a) * ARM_LENGTH * 0.82,
            cy + np.sin(a) * ARM_LENGTH * 0.82,
            cx + np.cos(a + s * SUB_BRANCH_ANGLE) * (ARM_LENGTH * 0.82 + SUB_BRANCH_LENGTH),
            cy + np.sin(a + s * SUB_BRANCH_ANGLE) * (ARM_LENGTH * 0.82 + SUB_BRANCH_LENGTH),
            SUB_BRANCH_THICKNESS
        )

for t in np.linspace(0, 2 * np.pi, 1000):
    for dx in range(-RING_THICKNESS, RING_THICKNESS + 1):
        for dy in range(-RING_THICKNESS, RING_THICKNESS + 1):
            mask.append((
                int(cx + np.cos(t) * INNER_RING_RADIUS + dx),
                int(cy + np.sin(t) * INNER_RING_RADIUS + dy)
            ))

mask = np.array(list(set(mask)), dtype=float)
mask[:, 0] = (mask[:, 0] // PIXEL_SIZE) * PIXEL_SIZE
mask[:, 1] = (mask[:, 1] // PIXEL_SIZE) * PIXEL_SIZE
mask = np.unique(mask, axis=0)

NUM = len(mask)

# ---------- PYGAME ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("❄️")
clock = pygame.time.Clock()

trail = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

particles = np.random.rand(NUM, 2)
particles[:, 0] *= WIDTH
particles[:, 1] *= HEIGHT

fuzz = np.random.uniform(-FUZZ_RADIUS, FUZZ_RADIUS, (NUM, 2))

scale = BASE_SCALE
running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            running = False

    bass_s = SMOOTHING * bass_s + (1 - SMOOTHING) * bass * BASS_GAIN

    trail.fill((8, 8, 14, 28))
    screen.blit(trail, (0, 0))

    particles += (mask - particles) * FLOW_STRENGTH

    target_scale = BASE_SCALE + bass_s * BASS_SCALE
    scale += (target_scale - scale) * SCALE_SPRING
    scale *= SCALE_DAMPING + 0.02

    #brightness = int(np.clip(200 + bass_s * 120, 200, 255))

    # Nonlinear bass → white (creates contrast)
    mix = np.clip((bass_s - 0.18) * 1.8, 0, 1)
    mix = mix * mix   # emphasize separation (gamma curve)

    target_color = ICE_BLUE * (1 - mix) + PURE_WHITE * mix

    current_color += (target_color - current_color) * COLOR_SMOOTH
    color = tuple(np.clip(current_color, 120, 255).astype(int))


    for i in range(NUM):
        dx = mask[i, 0] - cx
        dy = mask[i, 1] - cy

        # ✅ geometry scaled BEFORE bass scaling
        sx = cx + dx * GLOBAL_SCALE * scale + fuzz[i, 0]
        sy = cy + dy * GLOBAL_SCALE * scale + fuzz[i, 1]

        px = int(sx // PIXEL_SIZE) * PIXEL_SIZE
        py = int(sy // PIXEL_SIZE) * PIXEL_SIZE

        pygame.draw.rect(
            trail,
            (*color, 32),
            (px - 1, py - 1, PIXEL_SIZE + 2, PIXEL_SIZE + 2)
        )

        pygame.draw.rect(
            screen,
            color,
            (px, py, PIXEL_SIZE, PIXEL_SIZE)
        )


    pygame.display.flip()
    clock.tick(FPS)

stream.stop()
pygame.quit()
