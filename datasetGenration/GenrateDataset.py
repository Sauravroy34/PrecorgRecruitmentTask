from trdg.generators import GeneratorFromStrings
from PIL import Image
import os 
import random
import shutil
import numpy as np 

# --- CONFIGURATION ---
OUTPUT_DIR = "dataset_task2"
WORDS_FILE = "/home/saurav/Desktop/PrecorgTask/words.txt" # Make sure this points to your Task 2 list
FONTS_DIR = "/home/saurav/Desktop/PrecorgTask/available_fonts"
NUM_SAMPLES = 60000
ASSETS_DIR = "temp_assets" 

# Create directories
for subset in ["easy", "hard", "bonus"]:
    os.makedirs(f"{OUTPUT_DIR}/{subset}", exist_ok=True)

# --- 1. LOAD AND FILTER WORDS ---
print(f"Loading words from {WORDS_FILE}...")
with open(WORDS_FILE) as f:
    # 1. Strip whitespace
    # 2. Check length > 3
    # 3. Check isalnum() -> Returns True only if all characters are Alphabet or Numeric (No symbols)
    words = [
        w.strip() for w in f.readlines() 
        if len(w.strip()) > 3 and w.strip().isalnum()
    ]
    
print(f"Number of valid alphanumeric words loaded: {len(words)}")

# --- HELPERS ---
def get_random_words(count):
    return random.choices(words, k=count)

def randomize_case(word):
    return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in word)

def create_bg_asset(color, path, add_noise=False):
    """Creates a solid or noisy color image for TRDG to use as background"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if add_noise:
        arr = np.full((500, 500, 3), color, dtype=np.int16)
        noise = np.random.normal(0, 30, (500, 500, 3)).astype(np.int16)
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_arr)
    else:
        img = Image.new("RGB", (500, 500), color)
    img.save(path)

# ==========================================
# 1. EASY SET
# ==========================================
print("Generating Easy Set...")
easysetwords = [w.capitalize() for w in get_random_words(NUM_SAMPLES)]
labels_easy = []

easy_gen = GeneratorFromStrings(
    strings=easysetwords,
    count=NUM_SAMPLES,
    fonts=[f"{FONTS_DIR}/DejaVuMathTeXGyre.ttf"], 
    background_type=1, 
    blur=0, 
    random_skew=False,
    random_blur=False
)

for i, (img, lbl) in enumerate(easy_gen):
    filename = f"easy_{i}_.png"
    img.save(f"{OUTPUT_DIR}/easy/{filename}")
    labels_easy.append(f"{filename},{lbl},easy")

with open(f"{OUTPUT_DIR}/easy/labels.csv", "w") as f:
    f.write("\n".join(labels_easy))


# ==========================================
# 2. HARD SET
# ==========================================
print("Generating Hard Set...")
hard_raw_words = get_random_words(NUM_SAMPLES)
hard_words = [randomize_case(w) for w in hard_raw_words]
hard_fonts = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")]
labels_hard = []

hard_gen = GeneratorFromStrings(
    strings=hard_words,
    count=NUM_SAMPLES,
    fonts=hard_fonts,
    background_type=0,  # 0 = Gaussian Noise
    distorsion_type=3,  # Random distortion
    skewing_angle=15,   
    random_skew=True,
    text_color="#000000,#888888"
)

for i, (img, lbl) in enumerate(hard_gen):
    filename = f"hard_{i}.png"
    img.save(f"{OUTPUT_DIR}/hard/{filename}")
    labels_hard.append(f"{filename},{lbl},hard")

with open(f"{OUTPUT_DIR}/hard/labels.csv", "w") as f:
    f.write("\n".join(labels_hard))


# ==========================================
# 3. BONUS SET
# ==========================================
print("Generating Bonus Set...")

green_dir = os.path.join(ASSETS_DIR, "green")
red_dir = os.path.join(ASSETS_DIR, "red")

create_bg_asset((0, 255, 0), f"{green_dir}/bg.png", add_noise=True) 
create_bg_asset((255, 0, 0), f"{red_dir}/bg.png", add_noise=True)   

bonus_raw_words = get_random_words(NUM_SAMPLES)
green_words = []
red_words_input = [] 
labels_bonus = []

for w in bonus_raw_words:
    w_case = randomize_case(w)
    if random.random() > 0.5:
        green_words.append(w_case)
    else:
        # Reverse text for Red images
        red_words_input.append(w_case[::-1])

# Generator for GREEN (Normal)
if green_words:
    gen_green = GeneratorFromStrings(
        strings=green_words,
        count=len(green_words),
        fonts=hard_fonts,
        background_type=3,      
        image_dir=green_dir,    
        distorsion_type=3,
        skewing_angle=15,
        random_skew=True,
        text_color="#000000,#555555",
    )
    for i, (img, lbl) in enumerate(gen_green):
        filename = f"bonus_green_{i}.png"
        img.save(f"{OUTPUT_DIR}/bonus/{filename}")
        labels_bonus.append(f"{filename},{lbl},green")

# Generator for RED (Reversed)
if red_words_input:
    gen_red = GeneratorFromStrings(
        strings=red_words_input,
        count=len(red_words_input),
        fonts=hard_fonts,
        background_type=3,      
        image_dir=red_dir,      
        distorsion_type=3,
        skewing_angle=15,
        random_skew=True,
        text_color="#000000,#555555",
    )
    for i, (img, lbl) in enumerate(gen_red):
        original_label = lbl[::-1]
        filename = f"bonus_red_{i}.png"
        img.save(f"{OUTPUT_DIR}/bonus/{filename}")
        labels_bonus.append(f"{filename},{original_label},red")

with open(f"{OUTPUT_DIR}/bonus/labels.csv", "w") as f:
    f.write("\n".join(labels_bonus))

try:
    shutil.rmtree(ASSETS_DIR)
except:
    pass

print("Task 2 Dataset Generation Complete.")