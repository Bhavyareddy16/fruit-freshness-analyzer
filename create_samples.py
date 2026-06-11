import os
import cv2
import numpy as np

# Create sample_images directory
os.makedirs("sample_images", exist_ok=True)

def create_apple_fresh():
    # Create a blank 400x400 white image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 245
    
    # Draw stem (brown)
    cv2.line(img, (200, 120), (220, 70), (42, 77, 107), 6, lineType=cv2.LINE_AA)
    
    # Draw leaf (green)
    pts = np.array([[220, 70], [250, 60], [270, 80], [240, 90]], np.int32)
    cv2.fillPoly(img, [pts], (76, 177, 34))
    
    # Draw shiny red apple body (overlapping circles for heart-like shape)
    cv2.circle(img, (170, 210), 90, (35, 30, 220), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (230, 210), 90, (35, 30, 220), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (200, 240), 95, (35, 30, 220), -1, lineType=cv2.LINE_AA)
    
    # Draw shiny highlight (white curve)
    cv2.ellipse(img, (150, 170), (40, 20), 45, 0, 180, (255, 255, 255), 8, lineType=cv2.LINE_AA)
    
    # Add text label
    cv2.putText(img, "Fresh Apple", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2, lineType=cv2.LINE_AA)
    
    cv2.imwrite("sample_images/apple_fresh.png", img)

def create_apple_spoiled():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 245
    
    # Decay color (brownish dark red)
    decay_red = (25, 45, 110)
    stem_color = (25, 40, 55)
    
    # Draw stem
    cv2.line(img, (200, 120), (215, 80), stem_color, 6, lineType=cv2.LINE_AA)
    
    # Draw apple body (deformed, shriveled shape)
    cv2.circle(img, (175, 215), 85, decay_red, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (225, 215), 82, decay_red, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (200, 245), 90, decay_red, -1, lineType=cv2.LINE_AA)
    
    # Draw shriveled lines/creases (darker brown lines)
    cv2.ellipse(img, (160, 220), (35, 15), -30, 0, 360, (10, 20, 60), 2, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (240, 230), (25, 20), 20, 0, 360, (10, 20, 60), 2, lineType=cv2.LINE_AA)
    
    # Draw large soft rot spot (yellowish brown center, dark edge)
    cv2.circle(img, (220, 260), 35, (40, 80, 120), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (220, 260), 25, (60, 110, 160), -1, lineType=cv2.LINE_AA)
    
    # Mold spots (white/grey fuzzy circles)
    cv2.circle(img, (215, 255), 6, (200, 200, 200), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (230, 265), 4, (220, 220, 220), -1, lineType=cv2.LINE_AA)
    
    cv2.putText(img, "Spoiled Apple", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2, lineType=cv2.LINE_AA)
    
    cv2.imwrite("sample_images/apple_spoiled.png", img)

def create_banana_fresh():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 245
    
    # Draw curved yellow banana (using overlapping arcs or thick curve lines)
    # Using thick yellow line with green ends
    # We will draw a thick curve using ellipse outline
    ellipse_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.ellipse(ellipse_img, (220, 180), (120, 100), 25, 30, 150, (50, 220, 240), 50, lineType=cv2.LINE_AA)
    
    # Mask to keep the banana inside our image
    img[ellipse_img > 0] = ellipse_img[ellipse_img > 0]
    
    # Draw green ends (stem and tip)
    cv2.circle(img, (105, 240), 22, (50, 160, 70), -1, lineType=cv2.LINE_AA) # tip
    cv2.circle(img, (315, 145), 20, (30, 120, 50), -1, lineType=cv2.LINE_AA) # stem
    
    cv2.putText(img, "Fresh Banana", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2, lineType=cv2.LINE_AA)
    
    cv2.imwrite("sample_images/banana_fresh.png", img)

def create_banana_spoiled():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 245
    
    # Spoiled banana: dark brown/black with some yellow patches remaining
    decay_yellow = (20, 60, 100) # Brownish dark yellow
    decay_black = (20, 25, 35)   # Almost black
    
    ellipse_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.ellipse(ellipse_img, (220, 180), (120, 100), 25, 30, 150, decay_yellow, 50, lineType=cv2.LINE_AA)
    img[ellipse_img > 0] = ellipse_img[ellipse_img > 0]
    
    # Add large black rot patches
    cv2.circle(img, (150, 225), 25, decay_black, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (230, 180), 28, decay_black, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (280, 160), 22, decay_black, -1, lineType=cv2.LINE_AA)
    
    # Stem/tips
    cv2.circle(img, (105, 240), 22, (10, 30, 40), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (315, 145), 20, (10, 30, 40), -1, lineType=cv2.LINE_AA)
    
    # Tiny brown spots all over
    np.random.seed(99)
    for _ in range(30):
        x = np.random.randint(120, 300)
        y = np.random.randint(140, 250)
        # Verify if it's on the banana (not white background)
        if not np.all(img[y, x] == 245):
            cv2.circle(img, (x, y), np.random.randint(2, 6), decay_black, -1)
            
    cv2.putText(img, "Spoiled Banana", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2, lineType=cv2.LINE_AA)
    
    cv2.imwrite("sample_images/banana_spoiled.png", img)

def create_orange_fresh():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 245
    
    # Orange body (bright orange)
    cv2.circle(img, (200, 210), 100, (30, 130, 245), -1, lineType=cv2.LINE_AA)
    
    # Stem node (green center, brown dot)
    cv2.circle(img, (200, 115), 10, (50, 130, 60), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (200, 115), 3, (20, 60, 30), -1, lineType=cv2.LINE_AA)
    
    # Add orange texture (tiny yellow-orange dots)
    np.random.seed(50)
    for _ in range(120):
        x = np.random.randint(110, 290)
        y = np.random.randint(120, 300)
        if (x - 200)**2 + (y - 210)**2 < 95**2:
            cv2.circle(img, (x, y), 1, (50, 165, 255), -1)
            
    cv2.putText(img, "Fresh Orange", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2, lineType=cv2.LINE_AA)
    
    cv2.imwrite("sample_images/orange_fresh.png", img)

def create_orange_spoiled():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 245
    
    # Spoiled orange body (dull brown-orange)
    cv2.circle(img, (200, 210), 100, (40, 85, 160), -1, lineType=cv2.LINE_AA)
    
    # Shriveled outline overlay
    cv2.ellipse(img, (200, 210), (102, 98), 15, 0, 360, (20, 50, 110), 2, lineType=cv2.LINE_AA)
    
    # Large green-blue mold patch in center (Penicillium decay)
    mold_color = (130, 140, 80) # Greyish blue-green
    cv2.circle(img, (190, 200), 38, mold_color, -1, lineType=cv2.LINE_AA)
    
    # Fuzzy white mold border
    cv2.circle(img, (190, 200), 43, (215, 220, 210), 3, lineType=cv2.LINE_AA)
    
    # Another mold spot
    cv2.circle(img, (230, 230), 18, mold_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (230, 230), 21, (215, 220, 210), 2, lineType=cv2.LINE_AA)
    
    cv2.putText(img, "Spoiled Orange", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2, lineType=cv2.LINE_AA)
    
    cv2.imwrite("sample_images/orange_spoiled.png", img)

if __name__ == "__main__":
    create_apple_fresh()
    create_apple_spoiled()
    create_banana_fresh()
    create_banana_spoiled()
    create_orange_fresh()
    create_orange_spoiled()
    print("Created mock sample images inside sample_images/ directory.")
