import pygame
import sys

# Grid config
ROWS, COLS = 20, 40
CELL_SIZE = 30
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE

# Colors
BG_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
HIGHLIGHT_COLOR = (0, 200, 100)

# Init pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Grid Draw")
clock = pygame.time.Clock()

# Grid state: False = not highlighted, True = highlighted
grid = [[False for _ in range(COLS)] for _ in range(ROWS)]

def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = HIGHLIGHT_COLOR if grid[row][col] else BG_COLOR
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)  # grid lines

def get_cell(pos):
    x, y = pos
    col = x // CELL_SIZE
    row = y // CELL_SIZE
    if 0 <= row < ROWS and 0 <= col < COLS:
        return row, col
    return None

# Main loop
running = True
mouse_down = False
while running:
    screen.fill(BG_COLOR)
    draw_grid()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_down = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_down = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                highlighted = [(r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c]]
                print("Highlighted cells:")
                for r, c in highlighted:
                    print(f"Row {r}, Col {c}")
                print("-" * 30)

    if mouse_down:
        pos = pygame.mouse.get_pos()
        cell = get_cell(pos)
        if cell:
            r, c = cell
            grid[r][c] = True  # highlight on drag

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
