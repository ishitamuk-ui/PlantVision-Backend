"""Configuration constants and default parameters for DataSleuth."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    APP_NAME: str = "DataSleuth"
    TAGLINE: str = "Assess your dataset and compare ML performance, end to end."
    VERSION: str = "0.2.0"
    
    # File ingestion limits
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".csv",)
    DEFAULT_PREVIEW_ROWS: int = 15
    
    # Plotly performance thresholds
    MAX_SCATTER_POINTS: int = 5000
    CORRELATION_MAX_COLUMNS: int = 40
    
    # Quality Engine Thresholds
    IQR_MULTIPLIER: float = 1.5
    HIGH_MISSING_WARN_PCT: float = 20.0
    SEVERE_MISSING_WARN_PCT: float = 50.0
    HIGH_CARDINALITY_MAX_UNIQUE: int = 50
    HIGH_CARDINALITY_RATIO_WARN: float = 0.30
    CLASS_IMBALANCE_MINORITY_PCT: float = 15.0
    CLASS_IMBALANCE_RATIO_WARN: float = 4.0
    LEAKAGE_CORRELATION_THRESHOLD: float = 0.98
    
    # Editorial Dark & Lavender Visual Theme Tokens
    COLOR_BG_DARK: str = "#0B0A14"
    COLOR_CARD_BG_DARK: str = "#141224"
    COLOR_CARD_BORDER_DARK: str = "rgba(192, 132, 252, 0.14)"
    
    # Editorial Light & Lavender Visual Theme Tokens
    COLOR_BG_LIGHT: str = "#F8FAFC"
    COLOR_CARD_BG_LIGHT: str = "#FFFFFF"
    COLOR_CARD_BORDER_LIGHT: str = "rgba(226, 232, 240, 0.85)"
    
    COLOR_PRIMARY: str = "#C084FC"           # Muted Lavender (Dark)
    COLOR_PRIMARY_LIGHT: str = "#7C3AED"     # Rich Violet (Light)
    COLOR_SECONDARY: str = "#A855F7"         # Soft Violet
    COLOR_LAVENDER_BAR_DARK: str = "#B488F7" # Distribution bar color (Dark)
    COLOR_LAVENDER_BAR_LIGHT: str = "#7C3AED"# Distribution bar color (Light)
    COLOR_LAVENDER_LIGHT: str = "#DDD6FE"    # Pill background
    COLOR_SUCCESS: str = "#34D399"
    COLOR_WARNING: str = "#FBBF24"
    COLOR_DANGER: str = "#F87171"
    COLOR_TEXT_PRIMARY_DARK: str = "#F8FAFC"
    COLOR_TEXT_MUTED_DARK: str = "#94A3B8"
    COLOR_TEXT_PRIMARY_LIGHT: str = "#0F172A"
    COLOR_TEXT_MUTED_LIGHT: str = "#64748B"
    COLOR_PALETTE_DARK: tuple[str, ...] = (
        "#C084FC", "#818CF8", "#38BDF8", "#34D399", "#F472B6", "#A78BFA"
    )
    COLOR_PALETTE_LIGHT: tuple[str, ...] = (
        "#7C3AED", "#4F46E5", "#0284C7", "#059669", "#DB2777", "#9333EA"
    )
    # Legacy aliases
    COLOR_CARD_BG: str = "#141224"
    COLOR_CARD_BORDER: str = "#2B2644"
    COLOR_LAVENDER_BAR: str = "#B488F7"
    COLOR_TEXT_PRIMARY: str = "#F8FAFC"
    COLOR_TEXT_MUTED: str = "#94A3B8"
    COLOR_PALETTE: tuple[str, ...] = (
        "#C084FC", "#818CF8", "#38BDF8", "#34D399", "#F472B6", "#A78BFA"
    )


CONFIG = AppConfig()
