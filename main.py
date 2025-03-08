import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog
from typing import NamedTuple

# SELECT OPTION
# 1) CIE76, also known as the Euclidean Distance in the CIELAB color space
# 2) Weighted Euclidean Distance, where the weights (0.11, 0.59, and 0.3) correspond to the perceived sensitivity of the human eye to the red, green, and blue channels, respectively
# 3) Standard Euclidean Distance
OPTION = 1

# Load images
images = filedialog.askopenfilenames(
    title="Select Images", filetypes=(("png", "*.png"),)
)

class Perler(NamedTuple):
    name: str
    bgr: tuple[int, int, int]
    bgr_cielab: tuple[int, int, int]


# List of predefined Perler bead colors with their BGR and CIELAB values
Perlers = [
    Perler("White", (242, 247, 247), (248, 127, 130)),
    Perler("Light Grey", (191, 195, 190), (200, 125, 129)),
    Perler("Grey", (156, 152, 150), (160, 128, 126)),
    Perler("Pewter", (159, 161, 147), (166, 123, 127)),
    Perler("Charcoal", (95, 95, 84), (101, 124, 127)),
    Perler("Dark Grey", (92, 87, 86), (95, 129, 125)),
    Perler("Black", (52, 50, 52), (54, 130, 127)),
    Perler("Toasted Marshmallow", (216, 229, 241), (234, 130, 136)),
    Perler("Sand", (159, 196, 234), (208, 136, 152)),
    Perler("Fawn", (135, 176, 215), (190, 136, 154)),
    Perler("Tan", (137, 168, 207), (183, 138, 149)),
    Perler("Rust", (63, 78, 160), (110, 161, 153)),
    Perler("Cranapple", (79, 64, 136), (94, 161, 134)),
    Perler("Light Brown", (71, 123, 164), (139, 138, 162)),
    Perler("Gingerbread", (70, 84, 126), (102, 144, 143)),
    Perler("Brown", (77, 82, 108), (95, 138, 135)),
    Perler("Creme", (186, 231, 237), (232, 123, 151)),
    Perler("Pastel Yellow", (141, 238, 250), (238, 119, 176)),
    Perler("Yellow", (55, 215, 249), (221, 125, 205)),
    Perler("Cheddar", (78, 182, 255), (202, 145, 189)),
    Perler("Orange", (62, 128, 255), (172, 172, 184)),
    Perler("Butterscotch", (82, 154, 225), (177, 148, 176)),
    Perler("Honey", (44, 140, 218), (165, 150, 188)),
    Perler("Hot Coral", (88, 97, 255), (159, 187, 165)),
    Perler("Salmon", (127, 119, 255), (170, 180, 149)),
    Perler("Blush", (141, 158, 255), (190, 162, 152)),
    Perler("Flamingo", (190, 181, 255), (206, 156, 134)),
    Perler("Peach", (184, 198, 252), (215, 145, 142)),
    Perler("Light Pink", (213, 192, 245), (211, 150, 125)),
    Perler("Bubblegum", (157, 109, 225), (156, 178, 123)),
    Perler("Pink", (148, 87, 230), (148, 189, 124)),
    Perler("Magenta", (118, 70, 243), (145, 196, 140)),
    Perler("Fruit Punch", (89, 48, 218), (126, 194, 147)),
    Perler("Red", (68, 58, 196), (117, 183, 154)),
    Perler("Cherry", (69, 51, 173), (104, 178, 146)),
    Perler("Raspberry", (108, 60, 173), (110, 178, 125)),
    Perler("Plum", (170, 95, 178), (134, 172, 102)),
    Perler("Lavender", (211, 166, 180), (180, 142, 107)),
    Perler("Pastel Lavender", (187, 130, 149), (148, 147, 101)),
    Perler("Purple", (147, 84, 111), (104, 153, 97)),
    Perler("Blueberry Cream", (225, 167, 135), (174, 132, 95)),
    Perler("Periwinkle Blue", (191, 136, 108), (145, 133, 96)),
    Perler("Robin's Egg", (223, 217, 180), (215, 117, 121)),
    Perler("Pastel Blue", (214, 169, 99), (170, 119, 98)),
    Perler("Light Blue", (203, 138, 39), (140, 123, 87)),
    Perler("Cobalt", (179, 102, 0), (108, 133, 80)),
    Perler("Dark Blue", (124, 48, 43), (61, 152, 84)),
    Perler("Midnight", (70, 40, 22), (41, 132, 107)),
    Perler("Toothpaste", (213, 232, 176), (224, 106, 131)),
    Perler("Turquoise", (204, 143, 0), (143, 118, 88)),
    Perler("Light Green", (175, 199, 56), (186, 85, 129)),
    Perler("Parrot Green", (138, 150, 0), (142, 92, 125)),
    Perler("Pastel Green", (148, 213, 115), (199, 85, 151)),
    Perler("Kiwi Lime", (74, 202, 119), (189, 80, 182)),
    Perler("Bright Green", (96, 177, 84), (166, 83, 161)),
    Perler("Shamrock", (84, 150, 0), (139, 78, 154)),
    Perler("Dark Green", (85, 131, 16), (123, 87, 145)),
    Perler("Prickly Pear", (53, 215, 203), (211, 105, 201)),
    Perler("Evergreen", (79, 97, 60), (97, 110, 134)),
    Perler("Thistle", (175, 152, 153), (162, 133, 116)),
    Perler("Slime", (31, 206, 196), (203, 106, 203)),
    Perler("Mulberry", (104, 59, 109), (83, 157, 111)),
    Perler("Fuchsia", (177, 83, 221), (145, 192, 105)),
    Perler("Orange Cream", (145, 179, 255), (203, 152, 156)),
    Perler("Dark Spruce", (84, 74, 38), (75, 118, 118)),
    Perler("Denim", (153, 115, 78), (120, 126, 103)),
    Perler("Sage", (141, 189, 155), (187, 108, 149)),
    Perler("Slate Blue", (145, 133, 114), (139, 124, 119)),
    Perler("Sherbert", (125, 238, 225), (232, 108, 181)),
    Perler("Fern", (48, 151, 123), (149, 103, 177)),
    Perler("Olive", (62, 117, 115), (122, 119, 158)),
    Perler("Mist", (199, 185, 156), (187, 121, 118)),
    Perler("Sky", (227, 205, 84), (196, 101, 107)),
    Perler("Lagoon", (178, 171, 0), (162, 95, 114)),
    Perler("Apricot", (103, 169, 255), (195, 153, 174)),
    Perler("Orchid", (153, 108, 181), (140, 164, 116)),
    Perler("Spice", (68, 92, 227), (144, 179, 168)),
    Perler("Tomato", (66, 66, 234), (137, 192, 167)),
    Perler("Teal", (151, 141, 54), (138, 106, 115)),
    Perler("Rose", (114, 93, 210), (140, 176, 139)),
    Perler("Cotton Candy", (176, 121, 244), (170, 181, 121)),
    Perler("Eggplant", (85, 50, 111), (77, 159, 120)),
    Perler("Grape", (156, 59, 80), (83, 162, 78)),
    Perler("Tangerine", (24, 89, 253), (153, 188, 192)),
    Perler("Iris", (163, 86, 78), (101, 147, 85)),
    Perler("Forest ", (87, 93, 0), (89, 103, 125)),
    Perler("Sour Apple", (111, 222, 163), (210, 91, 176)),
    Perler("Mint", (213, 238, 179), (229, 104, 134)),
    Perler("Stone", (146, 152, 162), (162, 131, 132)),
    Perler("Cocoa", (70, 69, 80), (77, 133, 129)),
    Perler("Caribbean Sea", (158, 185, 0), (172, 82, 131)),
    Perler("Twilight Plum", (148, 117, 157), (138, 149, 117)),
    Perler("Frosted Lilac", (202, 192, 208), (202, 136, 125)),
]


for image in images:
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    new_img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    color_transform = {}  # Dictionary to store color transformation mapping
    new_color_count = {}  # Dictionary to count occurrences of each Perler color

    # Loop through all pixels in the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Skip transparent pixels (alpha != 255)
            if img[i][j][3] != 255:
                img[i][j] = [255, 255, 255, 0]  # Make transparent pixels white
                new_img[i][j] = [255, 255, 255, 0]  # Make transparent pixels white
                continue

            # Get the current pixel color in BGR format
            color = tuple(img[i, j])

            # If the color has been seen before, use the stored transformation
            if color in color_transform:
                index = color_transform[color]
                perler: Perler = Perlers[index]
                new_color_count[perler.name] += 1  # Increment count for the color
                new_img[i][j] = [perler.bgr[0], perler.bgr[1], perler.bgr[2], 255]
                continue

            # Convert the BGR color to CIELAB for color comparison
            colorLab = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2LAB)[0][0]

            # Initialize minimum distance for color matching
            min_d = float("inf")
            min_index = float("inf")

            # Compare the pixel color with each Perler color option
            for index, colors in enumerate(Perlers):
                match OPTION:
                    case 1:
                        # CIE76
                        d = (
                            math.pow(int(colorLab[0]) - int(colors.bgr_cielab[0]), 2)
                            + math.pow(int(colorLab[1]) - int(colors.bgr_cielab[1]), 2)
                            + math.pow(int(colorLab[2]) - int(colors.bgr_cielab[2]), 2)
                        )
                    case 2:
                        # Weighted Euclidean Distance
                        d = (
                            0.11 * math.pow(int(img[i][j][0]) - int(colors.bgr[0]), 2)
                            + 0.59 * math.pow(int(img[i][j][1]) - int(colors.bgr[1]), 2)
                            + 0.3 * math.pow(int(img[i][j][2]) - int(colors.bgr[2]), 2)
                        )
                    case 3:
                        # Standard Euclidean Distance
                        d = (
                            math.pow(int(img[i][j][0]) - int(colors.bgr[0]), 2)
                            + math.pow(int(img[i][j][1]) - int(colors.bgr[1]), 2)
                            + math.pow(int(img[i][j][2]) - int(colors.bgr[2]), 2)
                        )

                # If the current color is a better match, update the minimum distance
                if d < min_d:
                    min_d = d
                    min_index = index

            # Store the transformation for the current color
            color_transform[color] = min_index
            perler: Perler = Perlers[min_index]

            # Increment count for the selected Perler color
            new_color_count[perler.name] = new_color_count.get(perler.name, 0) + 1

            # Apply the new color to the transformed image
            new_img[i][j] = [perler.bgr[0], perler.bgr[1], perler.bgr[2], 255]

    # Print the count of each Perler color used in the image
    print("PERLER COUNTER")
    print("-----------------")
    for color in new_color_count:
        print(color + ": " + str(new_color_count[color]))
    print("\n\n\n")

    # Plot images
    plt.figure("Perler Counter")
    ax1 = plt.subplot(1, 2, 1)
    ax1.axis("off")  # Remove axes
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    ax1.set_title(
        "ORIGINAL COLORS\nAvailable Colors: 16,777,216\nColors Used: "
        + str(len(color_transform))
    )

    ax1 = plt.subplot(1, 2, 2)
    ax1.axis("off")  # Remove axes
    ax1.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    ax1.set_title(
        "MY COLORS\nAvailable Colors: "
        + str(len(Perlers))
        + "\nColors Used: "
        + str(len(new_color_count))
    )

    # Display the plot
    plt.show()
