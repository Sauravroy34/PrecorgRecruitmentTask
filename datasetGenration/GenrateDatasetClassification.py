from trdg.generators import GeneratorFromStrings
from PIL import Image
import os 
import random
import shutil
import numpy as np


OUTPUT_DIR = "dataset_task1"  
WORDS_FILE = "/home/saurav/Desktop/PrecorgTask/classification/google-10000-english-no-swears.txt"
FONTS_DIR = "/home/saurav/Desktop/PrecorgTask/available_fonts"
NUM_SAMPLES = 5000 
ASSETS_DIR = "temp_assets" 

# Create directories
for subset in ["easy", "hard", "bonus"]:
    os.makedirs(f"{OUTPUT_DIR}/{subset}", exist_ok=True)

print("Loading and splitting word lists...")
with open(WORDS_FILE) as f:
    all_words = [w.strip() for w in f.readlines() if len(w.strip()) > 3 and w.strip().isalpha()]

all_words = list(set(all_words))
random.shuffle(all_words)

task1_words = all_words[:100] 

print(f"Total words found: {len(all_words)}")
print(f"Selected {len(task1_words)} words for Task 1 Classification.")

with open("words_task1.txt", "w") as f:
    f.write("\n".join(task1_words))
    


def get_random_words(count):
    return random.choices(task1_words, k=count)

def randomize_case(word):
    return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in word)

def create_bg_asset(color, path, add_noise=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if add_noise:
        arr = np.full((500, 500, 3), color, dtype=np.int16)
        noise = np.random.normal(0, 30, (500, 500, 3)).astype(np.int16)
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_arr)
    else:
        img = Image.new("RGB", (500, 500), color)
    img.save(path)



print("Generating Easy Set (Task 1)...")
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
    filename = f"easy_{i}_{lbl}.png"
    img.save(f"{OUTPUT_DIR}/easy/{filename}")
    lbl = lbl.lower()
    labels_easy.append(f"{filename},{lbl},easy")

with open(f"{OUTPUT_DIR}/easy/labels.csv", "w") as f:
    f.write("\n".join(labels_easy))


print("Generating Hard Set (Task 1)...")
hard_raw_words = get_random_words(NUM_SAMPLES)
hard_words = [randomize_case(w) for w in hard_raw_words]
hard_fonts = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")]
labels_hard = []

hard_gen = GeneratorFromStrings(
    strings=hard_words,
    count=NUM_SAMPLES,
    fonts=hard_fonts,
    background_type=0, 
    distorsion_type=3, 
    skewing_angle=15,   
    random_skew=True,
    text_color="#000000,#888888"
)

for i, (img, lbl) in enumerate(hard_gen):
    filename = f"hard_{i}_{lbl}.png"
    img.save(f"{OUTPUT_DIR}/hard/{filename}")
    lbl = lbl.lower()
    labels_hard.append(f"{filename},{lbl},hard")

with open(f"{OUTPUT_DIR}/hard/labels.csv", "w") as f:
    f.write("\n".join(labels_hard))


print("Generating Bonus Set (Task 1)...")
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
        red_words_input.append(w_case[::-1])

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
        random_blur=True
    )
    for i, (img, lbl) in enumerate(gen_green):
        filename = f"bonus_green_{i}_{lbl}.png"
        img.save(f"{OUTPUT_DIR}/bonus/{filename}")
        lbl = lbl.lower()
        labels_bonus.append(f"{filename},{lbl},green")

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
        filename = f"bonus_red_{i}_{original_label}.png"
        img.save(f"{OUTPUT_DIR}/bonus/{filename}")
        original_label = original_label.lower()
        labels_bonus.append(f"{filename},{original_label},red")

with open(f"{OUTPUT_DIR}/bonus/labels.csv", "w") as f:
    f.write("\n".join(labels_bonus))

try:
    shutil.rmtree(ASSETS_DIR)
except:
    pass

print("Done! Task 1 Dataset (Classification) generated in 'dataset_task1'.")