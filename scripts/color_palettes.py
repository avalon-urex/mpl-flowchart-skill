"""
Universal Color Palettes and Theming for Publication-Quality Figures.
Supports Academic Light (pastel), Modern Dark, Minimal Grayscale, and Custom palettes.
"""

# 1. Academic Light (Default for IEEE, ACM, Nature, Science, CVPR, NeurIPS)
ACADEMIC_LIGHT = {
    "neutral": {"bg": "#F8F9FA", "border": "#868E96", "text": "#212529", "card_bg": "#FFFFFF", "card_border": "#CED4DA"},
    "blue":    {"bg": "#E7F5FF", "border": "#1C7ED6", "text": "#1864AB", "card_bg": "#FFFFFF", "card_border": "#1C7ED6", "tint": "#D0EBFF"},
    "purple":  {"bg": "#F3F0FF", "border": "#7048E8", "text": "#5F3DC4", "card_bg": "#FFFFFF", "card_border": "#7048E8", "tint": "#D0BFFF"},
    "green":   {"bg": "#EBFBEE", "border": "#2B8A3E", "text": "#237032", "card_bg": "#FFFFFF", "card_border": "#2B8A3E", "tint": "#D3F9D8"},
    "orange":  {"bg": "#FFF4E6", "border": "#D9480F", "text": "#C92A2A", "card_bg": "#FFFFFF", "card_border": "#D9480F", "tint": "#FFE8CC"},
    "yellow":  {"bg": "#FFF9DB", "border": "#F59F00", "text": "#D97706", "card_bg": "#FFFFFF", "card_border": "#F59F00", "tint": "#FFF3B0"},
    "pink":    {"bg": "#FFF0F6", "border": "#D6336C", "text": "#A61E4D", "card_bg": "#FFFFFF", "card_border": "#D6336C", "tint": "#FCC2D7"},
    "teal":    {"bg": "#E6FCF5", "border": "#0CA678", "text": "#087F5B", "card_bg": "#FFFFFF", "card_border": "#0CA678", "tint": "#C3FAE8"},
}

# 2. Modern Dark Mode (For presentations, GitHub READMEs, and technical blogs)
MODERN_DARK = {
    "canvas_bg": "#141517",
    "neutral": {"bg": "#1A1B1E", "border": "#5C5F66", "text": "#C1C2C5", "card_bg": "#25262B", "card_border": "#373A40"},
    "blue":    {"bg": "#182433", "border": "#339AF0", "text": "#74C0FC", "card_bg": "#1C2D42", "card_border": "#339AF0", "tint": "#103B66"},
    "purple":  {"bg": "#261C33", "border": "#845EF7", "text": "#B197FC", "card_bg": "#2F2142", "card_border": "#845EF7", "tint": "#422866"},
    "green":   {"bg": "#14291E", "border": "#40C057", "text": "#8CE99A", "card_bg": "#1A3828", "card_border": "#40C057", "tint": "#1E5235"},
    "orange":  {"bg": "#2D1D16", "border": "#F76707", "text": "#FFA94D", "card_bg": "#3B241A", "card_border": "#F76707", "tint": "#5E331E"},
    "yellow":  {"bg": "#2B2614", "border": "#FCC419", "text": "#FFE066", "card_bg": "#3B3317", "card_border": "#FCC419", "tint": "#5E4E19"},
    "pink":    {"bg": "#2E1724", "border": "#F06595", "text": "#FF8787", "card_bg": "#3B1D2D", "card_border": "#F06595", "tint": "#5E2543"},
    "teal":    {"bg": "#122826", "border": "#20C997", "text": "#63E6BE", "card_bg": "#163834", "card_border": "#20C997", "tint": "#1B544E"},
}

# 3. Minimal Grayscale (Camera-ready for strict black-and-white print criteria)
MINIMAL_MONO = {
    "neutral": {"bg": "#FAFAFA", "border": "#000000", "text": "#000000", "card_bg": "#FFFFFF", "card_border": "#000000"},
    "blue":    {"bg": "#F5F5F5", "border": "#222222", "text": "#000000", "card_bg": "#FFFFFF", "card_border": "#222222", "tint": "#E0E0E0"},
    "purple":  {"bg": "#EEEEEE", "border": "#333333", "text": "#000000", "card_bg": "#FFFFFF", "card_border": "#333333", "tint": "#D5D5D5"},
    "green":   {"bg": "#E8E8E8", "border": "#111111", "text": "#000000", "card_bg": "#FFFFFF", "card_border": "#111111", "tint": "#CCCCCC"},
    "orange":  {"bg": "#F0F0F0", "border": "#444444", "text": "#000000", "card_bg": "#FFFFFF", "card_border": "#444444", "tint": "#D8D8D8"},
}

THEMES = {
    "academic": ACADEMIC_LIGHT,
    "academic_light": ACADEMIC_LIGHT,
    "dark": MODERN_DARK,
    "modern_dark": MODERN_DARK,
    "mono": MINIMAL_MONO,
    "minimal_mono": MINIMAL_MONO,
}


def get_theme(theme_name="academic"):
    """Returns a theme dictionary by name."""
    if isinstance(theme_name, dict):
        return theme_name
    return THEMES.get(theme_name.lower(), ACADEMIC_LIGHT)
