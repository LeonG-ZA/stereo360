pragma Singleton
import QtQuick

QtObject {
    // Set from the window. A 1366x768 laptop leaves roughly 690 px of client
    // area once the title bar and taskbar are taken out, so on a short screen
    // the interface tightens its metrics instead of pushing the Convert
    // button off the bottom. Material sizes its controls from the font, so
    // the font steps below do most of the work.
    property bool compact: false

    // A dark surface by default: the app's main job is showing video frames,
    // and a bright chrome around a 360 still misleads you about its exposure.
    readonly property color bg:          "#0e1013"
    readonly property color surface:     "#171a20"
    readonly property color surfaceAlt:  "#1e222a"
    readonly property color border:      "#2a2f3a"
    readonly property color text:        "#e7eaf0"
    readonly property color textDim:     "#8b93a3"
    readonly property color textFaint:   "#5f6775"
    readonly property color accent:      "#6d8bff"
    readonly property color accentSoft:  "#2a3352"
    readonly property color warn:        "#f0a83c"
    readonly property color danger:      "#f0605d"
    readonly property color success:     "#4ec98a"

    readonly property int radius: compact ?  8 : 10
    readonly property int gap:    compact ? 10 : 14
    readonly property int pad:    compact ? 11 : 16

    readonly property int fontS:  compact ? 11 : 12
    readonly property int fontM:  compact ? 12 : 13
    readonly property int fontL:  compact ? 13 : 15
    readonly property int fontXL: compact ? 17 : 20

    readonly property int headerH: compact ? 44 : 58
    readonly property int footerH: compact ? 60 : 76
}
