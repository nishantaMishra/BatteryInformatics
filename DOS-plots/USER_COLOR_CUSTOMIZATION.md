# User Color Scheme Customization Guide

## Overview
The User color scheme allows you to define your own custom colors for elements in PDOS plots. This scheme is stored in the `ElementColorSchemes.yaml` file under the `User:` section and can be easily modified to suit your preferences.

## Usage
To use your custom color scheme:
```bash
# Use user-defined colors
Ti s p d, O p tot --colour user
--all -c custom fill
Ti d, O p -c user --grid
```

Both `user` and `custom` refer to the same customizable color scheme.

## How to Customize Colors

### 1. Locate the YAML File
The color definitions are stored in:
```
DOS-plots/PDOS/ElementColorSchemes.yaml
```

### 2. Find the User Section
Look for the `User:` section in the YAML file. It contains color definitions like:
```yaml
User:  # Customizable color scheme - modify these colors as needed
  H: [255, 255, 255]    # White
  C: [64, 64, 64]       # Dark Gray
  N: [0, 0, 255]        # Blue
  O: [255, 0, 0]        # Red
  # ... more elements
```

### 3. Modify Colors
Colors are defined as RGB values in the format `[Red, Green, Blue]` where each component ranges from 0-255.

**Examples of common colors:**
- White: `[255, 255, 255]`
- Black: `[0, 0, 0]`
- Red: `[255, 0, 0]`
- Green: `[0, 255, 0]`
- Blue: `[0, 0, 255]`
- Yellow: `[255, 255, 0]`
- Purple: `[128, 0, 128]`
- Orange: `[255, 165, 0]`
- Pink: `[255, 192, 203]`
- Gray: `[128, 128, 128]`

### 4. Add New Elements
To add colors for elements not currently defined, simply add new lines:
```yaml
User:
  # Existing colors...
  Pt: [211, 211, 211]  # Light Gray - your custom Platinum color
  Pd: [218, 165, 32]   # Golden Rod - your custom Palladium color
```

### 5. Save and Use
After modifying the YAML file:
1. Save the file
2. Run your PDOS plotting command with `-c user` or `--colour custom`
3. Your custom colors will be applied automatically

## Tips for Color Selection

### Color Harmony
- **Analogous colors**: Use colors next to each other on the color wheel
- **Complementary colors**: Use opposite colors for contrast
- **Monochromatic**: Use different shades of the same color

### Accessibility
- Ensure sufficient contrast between colors
- Consider colorblind-friendly palettes
- Avoid using only red/green combinations

### Scientific Conventions
- Keep similar elements in similar color families (e.g., all transition metals in blues)
- Use intuitive colors when possible (e.g., oxygen in red, nitrogen in blue)
- Maintain consistency with your other visualizations

### Online Color Tools
You can use online RGB color pickers to find exact RGB values:
- Google: Search "RGB color picker"
- Adobe Color: color.adobe.com
- Coolors.co: coolors.co

## Example Customizations

### High Contrast Theme
```yaml
User:
  H: [255, 255, 255]    # White
  C: [0, 0, 0]          # Black
  N: [0, 0, 255]        # Blue
  O: [255, 0, 0]        # Red
  Fe: [255, 165, 0]     # Orange
  Ti: [128, 0, 128]     # Purple
```

### Pastel Theme
```yaml
User:
  H: [255, 240, 245]    # Lavender Blush
  C: [211, 211, 211]    # Light Gray
  N: [173, 216, 230]    # Light Blue
  O: [255, 192, 203]    # Pink
  Fe: [255, 218, 185]   # Peach
  Ti: [221, 160, 221]   # Plum
```

### Materials-Focused Theme
```yaml
User:
  Ti: [70, 130, 180]    # Steel Blue (for titanium alloys)
  Fe: [139, 69, 19]     # Saddle Brown (for iron/rust)
  Al: [192, 192, 192]   # Silver (for aluminum)
  Cu: [184, 115, 51]    # Bronze/Copper
  Ni: [46, 125, 50]     # Dark Green
  O: [220, 20, 60]      # Crimson (for oxides)
```

## Troubleshooting

### Colors Not Updating
1. Check YAML syntax (proper indentation, colons, brackets)
2. Ensure RGB values are between 0-255
3. Restart the program to reload the color scheme

### YAML Syntax Errors
- Use spaces, not tabs for indentation
- Ensure proper formatting: `Element: [R, G, B]`
- Check that brackets and colons are present
- Element names must match exactly (case-sensitive)

### Missing Elements
- If an element isn't in your User scheme, the program will use default colors
- Add any missing elements you need to the User section

## Advanced Usage

### Backup Your Colors
Before making changes, consider backing up your custom colors:
```bash
cp ElementColorSchemes.yaml ElementColorSchemes_backup.yaml
```

### Share Color Schemes
You can share your User color scheme section with colleagues by copying the relevant YAML section.

### Multiple Custom Schemes
While only one User scheme is supported, you can create multiple versions by:
1. Saving different versions of the YAML file
2. Switching between them as needed
3. Or temporarily modifying the existing User section

---

**Need help?** The color scheme system automatically converts your RGB values to the format needed for plotting, so you just need to focus on choosing the right colors for your scientific visualization needs!