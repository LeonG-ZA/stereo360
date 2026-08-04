import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs
import Stereo360

ApplicationWindow {
    id: win
    // Sized for the smallest screen still in common use: 1366x768 leaves
    // about 690 px of client area after the title bar and taskbar. The
    // minimum goes lower still, because 768 at 125% scaling is only 614
    // logical pixels tall.
    width: 1180
    height: 660
    minimumWidth: 900
    minimumHeight: 560
    visible: true
    title: "stereo360"
    color: Theme.bg

    Material.theme: Material.Dark
    Material.accent: Theme.accent
    Material.foreground: Theme.text
    Material.background: Theme.surfaceAlt

    // Cascades to every control. Material sizes its controls from the font,
    // so this is also what keeps text fields and combo boxes from being tall
    // enough to push half the settings below the fold.
    font.pixelSize: Theme.fontM

    // Everything metric in the interface keys off this one measurement.
    Binding {
        target: Theme
        property: "compact"
        value: win.height < 750
    }

    // ---- settings state -------------------------------------------------
    property string inputPath: ""
    property string outputPath: ""
    property string outputMode: "360"
    property real yaw: 0             // only meaningful in vr180
    property int outputWidth: 0      // 0 = whatever the source implies
    property string quality: "standard"
    property string codec: ""        // "" = whatever the preset says
    property real strength: 1.0
    property real gradientLimit: 1.0
    property bool splitBaseline: false
    property bool spatialAudio: false
    property bool sourceSubsampling: false
    property bool faceSizeAuto: true
    property int faceSize: 1920
    property int depthTiles: 1
    property string depthBackend: "auto"
    property string depthModel: "base"
    property string onnxModel: ""
    property string device: "auto"
    property int fgErode: 2
    property int smooth: 0
    property string inpaint: "simple"
    property int chunkSize: 8
    property int chunkOverlap: 2
    property bool temporalFill: true
    property int startFrame: 0
    property int maxFrames: 0

    property bool logExpanded: true

    function currentOptions() {
        return {
            "input": inputPath, "output": outputPath, "quality": quality,
            "codec": codec, "outputMode": outputMode, "yaw": yaw,
            "outputWidth": outputWidth, "sourceWidth": sourceWidth,
            "strength": strength, "gradientLimit": gradientLimit,
            "splitBaseline": splitBaseline, "spatialAudio": spatialAudio,
            "sourceSubsampling": sourceSubsampling,
            "faceSizeAuto": faceSizeAuto, "faceSize": faceSize,
            "depthTiles": depthTiles, "depthBackend": depthBackend,
            "depthModel": depthModel, "onnxModel": onnxModel,
            "device": device, "fgErode": fgErode, "smooth": smooth,
            "inpaint": inpaint, "chunkSize": chunkSize,
            "chunkOverlap": chunkOverlap, "temporalFill": temporalFill,
            "startFrame": startFrame, "maxFrames": maxFrames
        }
    }

    readonly property bool canRun: inputPath !== "" && outputPath !== ""

    // Whatever set inputPath -- the dialog or the text field -- ask what it is.
    onInputPathChanged: {
        app.probeInput(inputPath)
        refreshThumbnail()
        // A width picked for an 8K file is not a legal choice for a 4K one,
        // and the CLI refuses to scale up rather than quietly obliging.
        outputWidth = 0
    }

    Component.onCompleted: app.probeBackends()

    // Encoder availability depends on the output size, and the output mode is
    // half of what decides that: the same source is 7680x7680 in 360 and
    // 7680x3840 in VR180, which is exactly where the hardware limits bite.
    function refreshEncoders() {
        if (sourceWidth > 0)
            app.probeEncoders(sourceWidth, app.sourceInfo.height,
                              outputMode, outputWidth)
    }

    // Only fetched for the mode that has a direction to choose. Requesting it
    // on the mode change as well as on the file means switching to VR180 finds
    // the picture already there.
    function refreshThumbnail() {
        if (outputMode === "vr180" && inputPath !== "")
            app.requestThumbnail(inputPath, previewFrame.value)
    }

    onOutputWidthChanged: refreshEncoders()

    onOutputModeChanged: {
        refreshEncoders()
        refreshThumbnail()
        // A yaw left over from a previous VR180 session would be silently
        // dropped by the 360 render and silently reappear on switching back.
        if (outputMode !== "vr180")
            yaw = 0
    }

    Connections {
        target: app
        function onSourceInfoChanged() { win.refreshEncoders() }
    }

    readonly property int sourceWidth:
        app.sourceInfo && app.sourceInfo.width ? app.sourceInfo.width : 0

    readonly property var resolutions:
        sourceWidth > 0
        ? app.resolutionChoices(sourceWidth, app.sourceInfo.height, outputMode)
        : []

    readonly property var outputSize:
        sourceWidth > 0
        ? app.outputSize(sourceWidth, app.sourceInfo.height, outputMode,
                         outputWidth)
        : null
    readonly property string outputSizeText:
        outputSize ? outputSize[0] + "×" + outputSize[1] : ""
    readonly property bool outputExceedsLevelCap:
        outputSize ? app.exceedsLevelLimit(outputSize[0], outputSize[1])
                   : false
    // 4, 9 or 16 channels is what ambiX looks like. Only a hint -- the file
    // does not say, which is why --spatial-audio exists at all -- but a hint
    // worth giving, because the cost of forgetting it is every sound at the
    // wrong bearing and nothing to hear that says so.
    readonly property bool sourceIsAmbisonic: {
        var n = app.sourceInfo ? app.sourceInfo.audio_channels : 0
        return n === 4 || n === 9 || n === 16
    }
    readonly property string outputMegapixels:
        outputSize ? (outputSize[0] * outputSize[1] / 1e6).toFixed(1) : ""

    // The dropdown's rows. Each says what it is and what it is for, because
    // "5760x5760" alone does not tell anyone which one they want.
    readonly property var resolutionModel: {
        var out = []
        for (var i = 0; i < resolutions.length; ++i) {
            var r = resolutions[i]
            out.push({
                width: r.width,
                text: r.label + (r.native ? "  ·  full size" : ""),
                sub: r.megapixels + " MP — "
                     + (r.fits ? "plays on a headset"
                               : "past the 35.6 MP decode limit; upload only")
            })
        }
        return out
    }
    readonly property int resolutionIndex: {
        var want = outputWidth === 0 ? sourceWidth : outputWidth
        for (var i = 0; i < resolutions.length; ++i)
            if (resolutions[i].width === want)
                return i
        return 0
    }

    function encoderEntry(name) {
        for (var i = 0; i < app.encoders.length; ++i)
            if (app.encoders[i].name === name)
                return app.encoders[i]
        return null
    }
    function encoderUsable(name) {
        if (name === "") return true                 // "from preset"
        var e = encoderEntry(name)
        return e === null || e.available             // unknown yet: allow
    }
    function encoderLabel(name) {
        if (name === "")
            return "The preset's own encoder"
        var e = encoderEntry(name)
        return e === null ? name : e.detail
    }

    function backendEntry(name) {
        for (var i = 0; i < app.backends.length; ++i)
            if (app.backends[i].name === name)
                return app.backends[i]
        return null
    }
    function backendUsable(name) {
        var e = backendEntry(name)
        return e === null || e.available     // unknown until probed: allow
    }
    function backendDetail(name) {
        var e = backendEntry(name)
        return e === null ? "" : e.detail
    }

    readonly property string sourceChroma:
        app.sourceInfo && app.sourceInfo.chroma ? app.sourceInfo.chroma : ""
    readonly property string sourceSummary:
        app.sourceInfo && app.sourceInfo.width
        ? app.sourceInfo.width + "×" + app.sourceInfo.height + " · "
          + app.sourceInfo.chroma + " · " + app.sourceInfo.frame_count
          + " frames"
        : ""

    readonly property var backendKeys: ["auto", "depth-anything",
                                        "video-depth-anything", "onnx"]
    function backendIndex(key) {
        return Math.max(0, backendKeys.indexOf(key))
    }

    // Built from the probe: "from preset" first, then whatever this machine
    // reported, hardware entries marked as the speed trade they are.
    readonly property var codecChoices: {
        var out = [{key: "", text: "From preset (recommended)", sub: ""}]
        for (var i = 0; i < app.encoders.length; ++i) {
            var e = app.encoders[i]
            out.push({key: e.name,
                      text: e.name + (e.hardware ? "  ·  not recommended" : ""),
                      sub: e.detail})
        }
        return out
    }
    function codecIndex(key) {
        for (var i = 0; i < codecChoices.length; ++i)
            if (codecChoices[i].key === key)
                return i
        return 0
    }
    // If the input changes to a size the chosen encoder cannot manage, fall
    // back rather than let the render fail on the first frame.
    onCodecChanged: if (!encoderUsable(codec)) codec = ""

    readonly property var qualityKeys: ["draft", "standard", "vr", "archival"]
    function qualityIndex(key) {
        return Math.max(0, qualityKeys.indexOf(key))
    }

    // The temporal backend ships small and large checkpoints only. Rather
    // than accept a Base selection and quietly run Small -- which is what it
    // did, so the dropdown said one thing and the render did another -- the
    // choice is simply not offered for that backend.
    readonly property var modelChoices: depthBackend === "video-depth-anything"
        ? [{key: "small", label: "Small — fastest"},
           {key: "large", label: "Large — largest, slowest"}]
        : [{key: "small", label: "Small — fastest"},
           {key: "base",  label: "Base — best quality per second"},
           {key: "large", label: "Large — largest, slowest"}]

    function modelIndex(key) {
        for (var i = 0; i < modelChoices.length; ++i)
            if (modelChoices[i].key === key)
                return i
        return 0
    }

    // Belt and braces behind the filtered list above, and deliberately not
    // expressed via modelChoices: a property changed handler can run before
    // the bindings that depend on the same property have re-evaluated, so
    // asking modelChoices here read the *previous* backend's list. Both
    // directions are checked explicitly instead, which is order-independent.
    function _clampModel() {
        if (depthBackend === "video-depth-anything" && depthModel === "base")
            depthModel = "small"
    }
    onDepthBackendChanged: _clampModel()
    onDepthModelChanged: _clampModel()

    // ---- dialogs --------------------------------------------------------
    FileDialog {
        id: openDialog
        title: "Choose a 360° video"
        nameFilters: ["Video files (*.mp4 *.mov *.mkv *.webm *.avi)",
                      "All files (*)"]
        onAccepted: {
            win.inputPath = app.toLocalPath(selectedFile.toString())
            if (win.outputPath === "")
                win.outputPath = app.suggestOutput(selectedFile.toString())
        }
    }

    FileDialog {
        id: onnxDialog
        title: "Choose an exported ONNX depth model"
        nameFilters: ["ONNX model (*.onnx)", "All files (*)"]
        onAccepted: win.onnxModel = app.toLocalPath(selectedFile.toString())
    }

    FileDialog {
        id: saveDialog
        title: "Save stereoscopic video as"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "mp4"
        nameFilters: ["MP4 video (*.mp4)"]
        onAccepted: win.outputPath = app.toLocalPath(selectedFile.toString())
    }

    // ---- log ------------------------------------------------------------
    ListModel { id: logModel }

    Connections {
        target: app
        function onLogged(level, text) {
            logModel.append({"level": level, "text": text})
            if (logModel.count > 500) logModel.remove(0, 100)
            // A collapsed log must never swallow a failure.
            if (level === "error") win.logExpanded = true
            logView.positionViewAtEnd()
        }
    }

    // =====================================================================
    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ---- header -----------------------------------------------------
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: Theme.headerH
            color: Theme.surface
            border.width: 0

            Rectangle {
                anchors.bottom: parent.bottom
                width: parent.width; height: 1
                color: Theme.border
            }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                spacing: 12

                Rectangle {
                    width: Theme.compact ? 22 : 26
                    height: width
                    radius: 7
                    color: Theme.accentSoft
                    border.width: 1
                    border.color: Theme.accent
                    Text {
                        anchors.centerIn: parent
                        text: "3D"
                        color: Theme.accent
                        font.pixelSize: 10
                        font.weight: Font.Bold
                    }
                }
                Text {
                    text: "stereo360"
                    color: Theme.text
                    font.pixelSize: Theme.fontXL
                    font.weight: Font.DemiBold
                }
                Text {
                    visible: !Theme.compact   // the title alone carries it
                    text: win.outputMode === "vr180"
                          ? "monoscopic 360° → stereoscopic VR180, side-by-side"
                          : "monoscopic 360° → stereoscopic 360°, top-bottom"
                    color: Theme.textFaint
                    font.pixelSize: Theme.fontS
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
                // Keeps the title left-aligned when the subtitle above is
                // hidden: without something that fills the width, a RowLayout
                // centres what is left.
                Item { Layout.fillWidth: true; visible: Theme.compact }
            }
        }

        // ---- body -------------------------------------------------------
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // ---- settings column ---------------------------------------
            Rectangle {
                Layout.preferredWidth: Theme.compact ? 408 : 466
                Layout.fillHeight: true
                color: Theme.bg

                ScrollView {
                    anchors.fill: parent
                    anchors.margins: Theme.gap
                    contentWidth: availableWidth
                    clip: true
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                    ColumnLayout {
                        width: parent.width
                        spacing: Theme.gap

                        // ---- files ---------------------------------------
                        Card {
                            title: "Files"

                            Row2 {
                                label: "Input"
                                TextField {
                                    Layout.fillWidth: true
                                    text: win.inputPath
                                    placeholderText: "Choose a video…"
                                    onEditingFinished: win.inputPath = text
                                }
                                Button {
                                    text: "Browse"
                                    onClicked: openDialog.open()
                                }
                            }

                            Row2 {
                                label: "Output"
                                TextField {
                                    Layout.fillWidth: true
                                    text: win.outputPath
                                    placeholderText: "Save as…"
                                    onEditingFinished: win.outputPath = text
                                }
                                Button {
                                    text: "Browse"
                                    onClicked: saveDialog.open()
                                }
                            }

                            // Here rather than further down because it is a
                            // property of the *source* audio, and because
                            // getting it wrong is not recoverable by
                            // re-tagging later -- it has to be decided
                            // before the render, not discovered after.
                            Row2 {
                                label: "Spatial audio"
                                hint: win.spatialAudio && win.outputMode === "vr180" && win.yaw !== 0
                                    ? "The soundfield will be turned " + win.yaw.toFixed(0) + "° to match the view, so sounds stay where you see them. The audio is re-encoded once to do it — the log names the codec — and the picture is unaffected."
                                    : win.outputMode === "vr180" && win.yaw !== 0
                                      ? (win.sourceIsAmbisonic
                                         ? "This source has " + app.sourceInfo.audio_channels + " audio channels, which is what ambiX looks like. Tick this and the soundfield turns with the view; leave it and every sound stays " + Math.abs(win.yaw).toFixed(0) + "° out of place."
                                         : "Tick this if the source audio really is ambiX: 4, 9 or 16 channels. Without it the view turns and the sound does not, leaving every source " + Math.abs(win.yaw).toFixed(0) + "° out of place.")
                                      : "Tick only if the source audio really is ambiX: 4, 9 or 16 channels. 360° and top-bottom are always written and need no setting."
                                Switch {
                                    checked: win.spatialAudio
                                    onToggled: win.spatialAudio = checked
                                }
                            }
                        }

                        // ---- output shape --------------------------------
                        Card {
                            title: "Output"
                            subtitle: "what kind of file to make"

                            Row2 {
                                label: "Format"
                                hint: win.outputMode === "vr180"
                                    ? "The middle 180°, eyes side by side. The same pixels spent on half the sphere, so twice the angular resolution — and the layout Apple Vision Pro content uses."
                                    : "A full sphere per eye, stacked top over bottom. Plays anywhere that plays 360° video."
                                ComboBox {
                                    id: modeBox
                                    Layout.fillWidth: true
                                    textRole: "label"
                                    valueRole: "key"
                                    currentIndex: win.outputMode === "vr180" ? 1 : 0
                                    model: [
                                        {key: "360",    label: "360 VR — top-bottom"},
                                        {key: "vr180",  label: "VR180 — side-by-side"}
                                    ]
                                    onActivated: win.outputMode = currentValue

                                    // Same as the quality preset: activating
                                    // the box severs its own binding, so a
                                    // later change to the property has to be
                                    // pushed back in by hand.
                                    Connections {
                                        target: win
                                        function onOutputModeChanged() {
                                            modeBox.currentIndex =
                                                win.outputMode === "vr180" ? 1 : 0
                                        }
                                    }
                                }
                                Text {
                                    text: win.outputSizeText
                                    color: Theme.textFaint
                                    font.pixelSize: Theme.fontS
                                    font.family: "Consolas, monospace"
                                }
                            }

                            // Only a choice when the source is big enough to
                            // give one. From a 4K file there is a single entry
                            // and nothing here to think about.
                            Row2 {
                                label: "Resolution"
                                visible: win.resolutions.length > 1
                                hint: win.outputWidth === 0
                                    ? "Full size — the right master for uploading, whatever it measures."
                                    : "Rendered at the source resolution and resized afterwards, so this is supersampled rather than rendered small. Costs the same time as full size."
                                ComboBox {
                                    id: resBox
                                    Layout.fillWidth: true
                                    textRole: "text"
                                    valueRole: "width"
                                    model: win.resolutionModel
                                    currentIndex: win.resolutionIndex
                                    onActivated: win.outputWidth =
                                        (currentValue === win.sourceWidth ? 0
                                                                          : currentValue)

                                    Connections {
                                        target: win
                                        function onOutputWidthChanged() {
                                            resBox.currentIndex = win.resolutionIndex
                                        }
                                    }

                                    delegate: ItemDelegate {
                                        width: resBox.width
                                        highlighted: resBox.highlightedIndex === index
                                        contentItem: ColumnLayout {
                                            spacing: 0
                                            Text {
                                                text: modelData.text
                                                color: Theme.text
                                                font.pixelSize: Theme.fontM
                                            }
                                            Text {
                                                visible: modelData.sub !== ""
                                                text: modelData.sub
                                                color: Theme.textFaint
                                                font.pixelSize: Theme.fontS
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                        }
                                    }
                                }
                            }

                            // A note, not a warning. Over the cap is the
                            // correct shape for a YouTube master, and saying
                            // otherwise would talk people out of the one
                            // workflow this tool is mostly used for. What it
                            // costs is direct playback, which is invisible
                            // until a headset refuses the file.
                            Rectangle {
                                Layout.fillWidth: true
                                visible: win.outputExceedsLevelCap
                                implicitHeight: levelNote.implicitHeight + 16
                                color: Theme.surfaceAlt
                                radius: 6
                                border.width: 1
                                border.color: Theme.border

                                Text {
                                    id: levelNote
                                    anchors.fill: parent
                                    anchors.margins: 8
                                    text: win.outputSizeText + " is "
                                        + win.outputMegapixels + " megapixels, "
                                        + "past the 35.6 that H.264 and HEVC "
                                        + "both cap at in their highest level. "
                                        + "Confirmed on a Quest 3: a file this "
                                        + "size loads and shows nothing, in "
                                        + "either codec. Keep it for uploading "
                                        + "— YouTube transcodes and this is the "
                                        + "right 8K 3D 360 master — and for "
                                        + "watching from a file, drop to "
                                        + (win.resolutions.length > 1
                                           ? win.resolutions[1].label : "a smaller size")
                                        + " above, or switch to VR180 at "
                                        + "full width."
                                    color: Theme.textDim
                                    font.pixelSize: Theme.fontS
                                    wrapMode: Text.WordWrap
                                }
                            }

                            DirectionPicker {
                                objectName: "directionPicker"
                                Layout.fillWidth: true
                                visible: win.outputMode === "vr180"
                                source: app.thumbnailSource
                                loading: win.inputPath !== ""
                                yaw: win.yaw
                                onYawMoved: (degrees) => win.yaw = degrees
                            }
                        }

                        // ---- quality -------------------------------------
                        Card {
                            title: "Encoding"
                            subtitle: "file size and compression only"

                            Row2 {
                                label: "Preset"
                                ComboBox {
                                    id: qualityBox
                                    Layout.fillWidth: true
                                    textRole: "label"
                                    valueRole: "key"
                                    currentIndex: win.qualityIndex(win.quality)
                                    model: [
                                        {key: "draft",    label: "Draft — x264 crf 20"},
                                        {key: "standard", label: "Standard — x264 crf 18"},
                                        {key: "vr",       label: "VR final — x265 crf 15 slow"},
                                        {key: "archival", label: "Archival — x265 crf 13 slow, 10-bit"}
                                    ]
                                    onActivated: win.quality = currentValue

                                    // ComboBox assigns currentIndex itself on
                                    // activation, which severs the binding
                                    // above -- so a later change to
                                    // win.quality would leave the box showing
                                    // one preset while the warning described
                                    // another. Re-apply it explicitly.
                                    Connections {
                                        target: win
                                        function onQualityChanged() {
                                            qualityBox.currentIndex =
                                                win.qualityIndex(win.quality)
                                        }
                                    }
                                }
                            }

                            Row2 {
                                label: "Subsampling"
                                hint: win.sourceChroma === "" ? ""
                                    : win.sourceChroma === "4:2:0"
                                      ? "The source is already 4:2:0, so this changes nothing."
                                      : "4:2:0 is the only layout headsets decode in hardware at 8K — untick this for anything you will play back directly. Measured on 4:2:0 footage, 4:4:4 was 4% more faithful for 25% more encode time."
                                CheckBox {
                                    text: "Use source subsampling"
                                    checked: win.sourceSubsampling
                                    onToggled: win.sourceSubsampling = checked
                                }
                                Rectangle {
                                    implicitWidth: chromaTag.implicitWidth + 14
                                    implicitHeight: chromaTag.implicitHeight + 8
                                    radius: 5
                                    color: Theme.surfaceAlt
                                    border.width: 1
                                    border.color: Theme.border
                                    Text {
                                        id: chromaTag
                                        anchors.centerIn: parent
                                        text: win.sourceChroma !== ""
                                              ? win.sourceChroma : "—"
                                        color: win.sourceChroma !== ""
                                               ? Theme.text : Theme.textFaint
                                        font.pixelSize: Theme.fontS
                                        font.family: "Consolas, monospace"
                                    }
                                }
                            }

                            Row2 {
                                label: "Encoder"
                                hint: win.codec === ""
                                      ? "The preset chooses a CPU encoder. Hardware encoders are faster but not equal quality per bit — they are a speed trade, not an upgrade."
                                      : win.encoderLabel(win.codec)
                                ComboBox {
                                    id: codecBox
                                    Layout.fillWidth: true
                                    textRole: "text"
                                    valueRole: "key"
                                    model: win.codecChoices
                                    currentIndex: win.codecIndex(win.codec)
                                    onActivated: {
                                        if (win.encoderUsable(currentValue))
                                            win.codec = currentValue
                                        else
                                            currentIndex = win.codecIndex(win.codec)
                                    }

                                    delegate: ItemDelegate {
                                        width: codecBox.width
                                        enabled: win.encoderUsable(modelData.key)
                                        highlighted:
                                            codecBox.highlightedIndex === index
                                        contentItem: ColumnLayout {
                                            spacing: 0
                                            Text {
                                                text: modelData.text
                                                color: enabled ? Theme.text
                                                               : Theme.textFaint
                                                font.pixelSize: Theme.fontM
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                            Text {
                                                visible: modelData.sub !== ""
                                                text: modelData.sub
                                                color: enabled ? Theme.textFaint
                                                               : Theme.warn
                                                font.pixelSize: Theme.fontS
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                        }
                                    }
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true
                                visible: noteText.text !== ""
                                implicitHeight: noteText.implicitHeight + 16
                                color: "#2a2114"
                                radius: 6
                                border.width: 1
                                border.color: Theme.warn

                                Text {
                                    id: noteText
                                    anchors.fill: parent
                                    anchors.margins: 8
                                    text: app.presetNote(win.quality)
                                    color: Theme.warn
                                    font.pixelSize: Theme.fontS
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }

                        // ---- 3D ------------------------------------------
                        Card {
                            title: "3D effect"
                            subtitle: "independent of the encoding preset"

                            Row2 {
                                label: "Strength"
                                hint: "Higher is a stronger stereo effect. 1.0 is a comfortable default."
                                Slider {
                                    Layout.fillWidth: true
                                    from: 0; to: 3; stepSize: 0.05
                                    value: win.strength
                                    onMoved: win.strength = value
                                }
                                Text {
                                    text: win.strength.toFixed(2)
                                    color: Theme.text
                                    font.pixelSize: Theme.fontM
                                    Layout.preferredWidth: 32
                                }
                            }

                            Row2 {
                                label: "Gradient limit"
                                hint: "Prevents holes rather than filling them. Raise it if thin structures look flat; 0 disables."
                                Slider {
                                    Layout.fillWidth: true
                                    from: 0; to: 3; stepSize: 0.05
                                    value: win.gradientLimit
                                    onMoved: win.gradientLimit = value
                                }
                                Text {
                                    text: win.gradientLimit.toFixed(2)
                                    color: Theme.text
                                    font.pixelSize: Theme.fontM
                                    Layout.preferredWidth: 32
                                }
                            }

                            Row2 {
                                label: "Split baseline"
                                hint: "Warp both eyes half as far in opposite directions. Far fewer holes per eye."
                                Switch {
                                    checked: win.splitBaseline
                                    onToggled: win.splitBaseline = checked
                                }
                            }

                        }

                        // ---- depth ---------------------------------------
                        Card {
                            title: "Depth"
                            subtitle: "independent of the encoding preset"

                            Row2 {
                                label: "Backend"
                                hint: win.backendDetail(win.depthBackend)
                                ComboBox {
                                    id: backendBox
                                    Layout.fillWidth: true
                                    model: ["auto", "depth-anything",
                                            "video-depth-anything", "onnx"]
                                    currentIndex: 0
                                    onActivated: {
                                        if (win.backendUsable(currentText))
                                            win.depthBackend = currentText
                                        else
                                            currentIndex = win.backendIndex(
                                                win.depthBackend)
                                    }

                                    // Unavailable backends stay listed but
                                    // cannot be chosen. Hiding them would
                                    // leave no clue they exist; letting them
                                    // be chosen just moves the failure later.
                                    delegate: ItemDelegate {
                                        width: backendBox.width
                                        enabled: win.backendUsable(modelData)
                                        highlighted:
                                            backendBox.highlightedIndex === index
                                        contentItem: ColumnLayout {
                                            spacing: 0
                                            Text {
                                                text: modelData
                                                color: enabled ? Theme.text
                                                               : Theme.textFaint
                                                font.pixelSize: Theme.fontM
                                            }
                                            Text {
                                                visible: !enabled
                                                text: win.backendDetail(modelData)
                                                color: Theme.warn
                                                font.pixelSize: Theme.fontS
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                        }
                                    }
                                }
                            }

                            // What "model" means depends on the backend, so
                            // only the row that applies is shown. That makes
                            // the relationship self-teaching: the ONNX path
                            // appears once the ONNX backend is selected, which
                            // is the only configuration that reads it.
                            Row2 {
                                label: "Model"
                                visible: win.depthBackend !== "onnx"
                                hint: win.depthModel === "base"
                                      ? "The default. Measured best on 8K footage: lowest depth noise and 40% less flicker than Small. Downloads ~400 MB on first use."
                                      : win.depthModel === "large"
                                        ? "Twice Base's extra cost and measured no better on 8K footage. Downloads ~1.3 GB on first use."
                                        : "The sharpest, and the noisiest — that noise is what makes thin structures shift between frames. About 9% faster overall than Base."
                                ComboBox {
                                    Layout.fillWidth: true
                                    textRole: "label"
                                    valueRole: "key"
                                    currentIndex: win.modelIndex(win.depthModel)
                                    model: win.modelChoices
                                    onActivated: win.depthModel = currentValue

                                    // Same reason as the quality preset: the
                                    // control severs its own binding when
                                    // activated, so re-apply it explicitly.
                                    Connections {
                                        target: win
                                        function onDepthModelChanged() {
                                            parent.currentIndex =
                                                win.modelIndex(win.depthModel)
                                        }
                                    }
                                }
                            }

                            Row2 {
                                label: "ONNX model"
                                visible: win.depthBackend === "onnx"
                                hint: "A graph from scripts/export_onnx.py. A smaller --size is the fast mode for slow machines: --size 266 measured 2.8× faster depth on a CPU at 0.989 depth correlation. Leave blank for the default model."
                                TextField {
                                    Layout.fillWidth: true
                                    text: win.onnxModel
                                    placeholderText: "models/depth_anything_v2_small.onnx"
                                    onEditingFinished: win.onnxModel = text
                                }
                                Button {
                                    text: "Browse"
                                    onClicked: onnxDialog.open()
                                }
                            }

                            Row2 {
                                label: "Device"
                                ComboBox {
                                    Layout.fillWidth: true
                                    model: ["auto", "cuda", "mps", "cpu"]
                                    currentIndex: 0
                                    onActivated: win.device = currentText
                                }
                            }

                            Row2 {
                                label: "Depth tiles"
                                hint: "N×N tiles per cube face. Finer depth on railings and cables; N² times slower."
                                SpinBox {
                                    from: 1; to: 4
                                    value: win.depthTiles
                                    onValueModified: win.depthTiles = value
                                }
                            }

                            Row2 {
                                label: "Face size"
                                hint: faceAuto.checked
                                      ? "Input width ÷ 4 — the lossless value."
                                      : "Lower than the auto value loses detail."
                                Switch {
                                    id: faceAuto
                                    text: "Auto"
                                    checked: win.faceSizeAuto
                                    onToggled: win.faceSizeAuto = checked
                                }
                                SpinBox {
                                    visible: !faceAuto.checked
                                    from: 128; to: 4096; stepSize: 64
                                    value: win.faceSize
                                    onValueModified: win.faceSize = value
                                }
                            }
                        }

                        // ---- range ---------------------------------------
                        Card {
                            title: "Frame range"
                            subtitle: "for test renders"

                            Row2 {
                                label: "Start at"
                                SpinBox {
                                    from: 0; to: 999999; stepSize: 30
                                    value: win.startFrame
                                    onValueModified: win.startFrame = value
                                }
                            }

                            Row2 {
                                label: "Limit"
                                hint: "0 renders to the end."
                                SpinBox {
                                    from: 0; to: 999999; stepSize: 30
                                    value: win.maxFrames
                                    onValueModified: win.maxFrames = value
                                }
                            }
                        }

                        // ---- advanced ------------------------------------
                        Card {
                            title: "Advanced"
                            subtitle: "rarely needed"
                            collapsible: true
                            expanded: false

                            Row2 {
                                label: "Edge erode"
                                hint: "Pixels of foreground erosion at depth edges."
                                SpinBox {
                                    from: 0; to: 16
                                    value: win.fgErode
                                    onValueModified: win.fgErode = value
                                }
                            }

                            Row2 {
                                label: "Depth smooth"
                                hint: "Guided-filter radius. Off by default — it cost 66% of runtime for no visible gain."
                                SpinBox {
                                    from: 0; to: 32
                                    value: win.smooth
                                    onValueModified: win.smooth = value
                                }
                            }

                            Row2 {
                                label: "Inpaint"
                                ComboBox {
                                    Layout.fillWidth: true
                                    model: ["simple", "learned"]
                                    onActivated: win.inpaint = currentText
                                }
                            }

                            Row2 {
                                label: "Chunk size"
                                hint: "Temporal context length. Only used by video-depth-anything."
                                enabled: win.depthBackend === "video-depth-anything"
                                SpinBox {
                                    from: 1; to: 32
                                    value: win.chunkSize
                                    onValueModified: win.chunkSize = value
                                }
                            }

                            Row2 {
                                label: "Chunk overlap"
                                enabled: win.depthBackend === "video-depth-anything"
                                SpinBox {
                                    from: 0; to: 16
                                    value: win.chunkOverlap
                                    onValueModified: win.chunkOverlap = value
                                }
                            }

                            Row2 {
                                label: "Temporal fill"
                                enabled: win.depthBackend === "video-depth-anything"
                                Switch {
                                    checked: win.temporalFill
                                    onToggled: win.temporalFill = checked
                                }
                            }
                        }

                        Item { Layout.preferredHeight: 4 }
                    }
                }
            }

            Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true
                        color: Theme.border }

            // ---- preview + log column ----------------------------------
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.margins: Theme.gap
                spacing: Theme.gap

                // ---- preview -------------------------------------------
                Rectangle {
                    id: previewPanel
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: Theme.compact ? 170 : 260
                    color: Theme.surface
                    radius: Theme.radius
                    border.width: 1
                    border.color: Theme.border
                    clip: true

                    // Bound to the controller string, not to Image.source:
                    // source is a url, and a url compared against "" is not
                    // false the way a string is, so the placeholder and the
                    // eye labels both showed with nothing rendered.
                    readonly property bool hasPreview: app.previewSource !== ""

                    Image {
                        id: previewImage
                        anchors.fill: parent
                        anchors.margins: 1
                        source: app.previewSource
                        fillMode: Image.PreserveAspectFit
                        asynchronous: true
                        cache: false
                        visible: parent.hasPreview
                    }

                    ColumnLayout {
                        anchors.centerIn: parent
                        spacing: 6
                        visible: !parent.hasPreview
                        Text {
                            text: "No preview yet"
                            color: Theme.textDim
                            font.pixelSize: Theme.fontL
                            Layout.alignment: Qt.AlignHCenter
                        }
                        Text {
                            text: "Render one frame to judge strength and depth\n" +
                                  "before committing to the whole video."
                            color: Theme.textFaint
                            font.pixelSize: Theme.fontS
                            horizontalAlignment: Text.AlignHCenter
                            Layout.alignment: Qt.AlignHCenter
                        }
                    }

                    // Eye labels, placed against the *painted* image rather
                    // than the panel. The output is square and the panel is
                    // wide, so PreserveAspectFit leaves broad letterbox bars
                    // — labels pinned to the panel would sit in empty space
                    // beside the picture they are naming. They also follow
                    // the packing: VR180 puts the eyes side by side, and a
                    // label reading "right eye" over the left one would be
                    // worse than no label at all.
                    readonly property real paintedW: previewImage.paintedWidth
                    readonly property real paintedH: previewImage.paintedHeight
                    readonly property real paintedX:
                        (width - paintedW) / 2
                    readonly property real paintedY:
                        (height - paintedH) / 2

                    Repeater {
                        model: parent.hasPreview ? ["Left eye", "Right eye"] : []
                        Rectangle {
                            readonly property bool sideBySide:
                                win.outputMode === "vr180"
                            x: previewPanel.paintedX + 10
                               + (sideBySide
                                  ? index * previewPanel.paintedW / 2 : 0)
                            y: previewPanel.paintedY + 10
                               + (sideBySide
                                  ? 0 : index * previewPanel.paintedH / 2)
                            width: tag.implicitWidth + 14
                            height: 22
                            radius: 5
                            color: "#c0000000"
                            Text {
                                id: tag
                                anchors.centerIn: parent
                                text: modelData
                                color: Theme.text
                                font.pixelSize: Theme.fontS
                            }
                        }
                    }

                    BusyIndicator {
                        anchors.centerIn: parent
                        running: app.busy && app.previewMode
                        visible: running
                    }
                }

                // ---- preview controls ----------------------------------
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10

                    Text {
                        text: "Preview frame"
                        color: Theme.textDim
                        font.pixelSize: Theme.fontM
                    }
                    SpinBox {
                        id: previewFrame
                        from: 0
                        to: app.sourceInfo && app.sourceInfo.frame_count
                            ? Math.max(0, app.sourceInfo.frame_count - 1)
                            : 999999
                        stepSize: 15
                        value: 0
                        // The direction picker drags on this frame, so it has
                        // to follow the same one the preview would render.
                        onValueModified: win.refreshThumbnail()
                    }
                    Text {
                        text: win.sourceSummary
                        color: Theme.textFaint
                        font.pixelSize: Theme.fontS
                        elide: Text.ElideRight
                        Layout.fillWidth: true
                    }
                    Button {
                        text: "Render preview"
                        enabled: win.inputPath !== "" && !app.busy
                        onClicked: app.preview(win.currentOptions(),
                                               previewFrame.value)
                    }
                }

                // ---- log -----------------------------------------------
                // Collapsible, because on a short screen this panel is the
                // one thing worth trading away for a bigger preview. An
                // error re-opens it (see onLogged) so collapsing it can never
                // hide a failure.
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: win.logExpanded
                                            ? (Theme.compact ? 98 : 150)
                                            : logHeader.height + 12
                    color: Theme.surface
                    radius: Theme.radius
                    border.width: 1
                    border.color: Theme.border
                    clip: true

                    Behavior on Layout.preferredHeight {
                        NumberAnimation { duration: 120
                                          easing.type: Easing.OutCubic }
                    }

                    Item {
                        id: logHeader
                        anchors.top: parent.top
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: 6
                        height: 20

                        Text {
                            anchors.left: parent.left
                            anchors.leftMargin: 4
                            anchors.verticalCenter: parent.verticalCenter
                            text: "Log"
                            color: Theme.textDim
                            font.pixelSize: Theme.fontS
                            font.weight: Font.DemiBold
                        }
                        Text {
                            anchors.right: parent.right
                            anchors.rightMargin: 4
                            anchors.verticalCenter: parent.verticalCenter
                            text: win.logExpanded ? "–" : "+"
                            color: Theme.textDim
                            font.pixelSize: Theme.fontL
                        }
                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: win.logExpanded = !win.logExpanded
                        }
                    }

                    ListView {
                        id: logView
                        anchors.top: logHeader.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        anchors.margins: 10
                        anchors.topMargin: 2
                        visible: win.logExpanded
                        clip: true
                        model: logModel
                        spacing: 2
                        ScrollBar.vertical: ScrollBar {}

                        delegate: Text {
                            width: logView.width - 12
                            text: model.text
                            wrapMode: Text.Wrap
                            font.family: "Consolas, monospace"
                            font.pixelSize: Theme.fontS
                            color: model.level === "error" ? Theme.danger
                                 : model.level === "warning" ? Theme.warn
                                 : Theme.textDim
                        }
                    }

                    Text {
                        anchors.centerIn: logView
                        visible: win.logExpanded && logModel.count === 0
                        text: "Output from stereo360 appears here"
                        color: Theme.textFaint
                        font.pixelSize: Theme.fontS
                    }
                }
            }
        }

        // ---- footer -----------------------------------------------------
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: Theme.footerH
            color: Theme.surface

            Rectangle {
                width: parent.width; height: 1
                color: Theme.border
            }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                spacing: 16

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    RowLayout {
                        spacing: 8
                        Text {
                            text: app.status
                            color: app.status === "Failed" ? Theme.danger
                                 : app.status === "Finished" ? Theme.success
                                 : Theme.text
                            font.pixelSize: Theme.fontM
                            font.weight: Font.DemiBold
                        }
                        Text {
                            text: app.detail
                            color: Theme.textFaint
                            font.pixelSize: Theme.fontS
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }
                    }

                    ProgressBar {
                        Layout.fillWidth: true
                        from: 0; to: 1
                        value: app.progress
                    }
                }

                Button {
                    text: "Stop"
                    enabled: app.busy
                    onClicked: app.cancel()
                }

                Button {
                    text: "Convert"
                    highlighted: true
                    enabled: win.canRun && !app.busy
                    onClicked: {
                        logModel.clear()
                        app.convert(win.currentOptions())
                    }
                }
            }
        }
    }
}
