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
    // The last name this window proposed. Kept so it can tell a suggestion it
    // is free to revise from a path the person typed or picked themselves.
    property string suggestedOutput: ""
    property string outputMode: "360"
    property real yaw: 0             // only meaningful in vr180
    property int outputWidth: 0      // 0 = whatever the source implies
    // The width this window chose on the source's behalf, so it can tell its
    // own suggestion from a size someone picked. Same reasoning as
    // suggestedOutput above.
    property int suggestedOutputWidth: 0
    property string quality: "standard"
    property string codec: ""        // "" = whatever the preset says
    property real strength: 1.0
    property real gradientLimit: 1.0
    property real faceAngularCorrection: 0.0
    // Two controls the user thinks in -- which eye stays sharp, and how
    // much of the separation it carries -- and the single number the
    // CLI takes. 0 leaves the left eye untouched, 1 the right, 0.5 is
    // an even split.
    property bool sourceRight: false
    property real baselineShare: 0.0
    readonly property real leftShare: sourceRight ? 1.0 - baselineShare : baselineShare
    property bool spatialAudio: false
    property bool sourceSubsampling: false
    property bool faceSizeAuto: true
    property int faceSize: 1920
    property int depthTiles: 1
    // Empty means "let the CLI choose for this kind of job" -- V3 for video,
    // Depth Pro for a still. Same sentinel as the encoder's "from preset":
    // the flag is simply not passed, so the two stay in step by construction
    // rather than by the UI knowing what the defaults are.
    property string depthBackend: ""
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
            "faceAngularCorrection": faceAngularCorrection,
            "leftShare": leftShare, "spatialAudio": spatialAudio,
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

    // Refresh the Output box for `fileUrl`. The rule for when a name may be
    // replaced lives in options.resolve_output, where it can be tested --
    // this stays a call so the two cannot say different things.
    function adoptSuggestedOutput(fileUrl) {
        var r = app.resolveOutput(win.outputPath, win.suggestedOutput,
                                  fileUrl, win.outputMode)
        win.outputPath = r.output
        win.suggestedOutput = r.suggested
    }

    // A photo job. Most of this window is about video, and showing controls
    // that do nothing implies they do something -- so the ones that cannot
    // apply are hidden rather than disabled. The CLI decides the same way,
    // from the input's extension, so the two cannot disagree.
    readonly property bool photoMode: inputPath !== "" && app.isImage(inputPath)

    // Whatever set inputPath -- the dialog or the text field -- ask what it is.
    onInputPathChanged: {
        app.probeInput(inputPath)
        refreshThumbnail()
        // A width picked for an 8K file is not a legal choice for a 4K one,
        // and the CLI refuses to scale up rather than quietly obliging.
        outputWidth = 0
    }

    Component.onCompleted: app.probeBackends()

    // Start a big 360 source at a size that plays, rather than at full size.
    // Resolved to a real number the moment the probe lands, never left as a
    // sentinel: the box, the encoder probe and the command line all read
    // `outputWidth`, and a "0 means work it out" that each resolved for
    // itself is how this control once displayed one size and rendered
    // another.
    //
    // Reads `app` rather than the `sourceWidth` and `photoMode` bindings,
    // because this runs from a property-changed handler and those may not
    // have re-evaluated yet -- the trap documented on _clampModel.
    //
    // Only revises a width this window chose. A size picked by hand survives
    // a mode switch.
    function adoptDefaultResolution() {
        if (!app.sourceInfo || !app.sourceInfo.width)
            return
        var w = app.defaultOutputWidth(app.sourceInfo.width,
                                       app.sourceInfo.height, outputMode,
                                       app.isImage(inputPath))
        if (outputWidth === 0 || outputWidth === suggestedOutputWidth)
            outputWidth = w
        suggestedOutputWidth = w
    }

    // The CLI uses more tiles for a photo than for a video, so the spin box
    // has to follow -- otherwise it reads 1 while the render does 3, which is
    // the same "shows one thing, does another" failure the resolution box
    // already had twice. Only revises a value this window chose.
    property int suggestedDepthTiles: 1
    function adoptPhotoDefaults() {
        var want = app.isImage(inputPath) ? app.photoDepthTiles : 1
        if (depthTiles === suggestedDepthTiles)
            depthTiles = want
        suggestedDepthTiles = want
    }

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
    // Wanted in two places now: the VR180 direction picker drags on it, and
    // in photo mode the panel shows it as the source before conversion.
    function refreshThumbnail() {
        if (inputPath === "")
            return
        // Asks `app` directly rather than reading the `photoMode` binding.
        // This runs from onInputPathChanged, and a change handler can fire
        // before the bindings depending on the same property have
        // re-evaluated -- so photoMode would still describe the *previous*
        // file. The same trap is documented on _clampModel below; here it
        // cost a panel that stayed on "Opening the photo..." forever.
        var photo = app.isImage(inputPath)
        if (photo || outputMode === "vr180")
            app.requestThumbnail(inputPath, photo ? 0 : previewFrame.value)
    }

    onOutputWidthChanged: refreshEncoders()

    onOutputModeChanged: {
        // The cap only bites in 360: the same source is 7680x7680 there and
        // 7680x3840 in VR180, which plays. So the right default moves with
        // the mode, and switching back should not leave a reduction behind
        // that only the other mode needed.
        adoptDefaultResolution()
        refreshEncoders()
        refreshThumbnail()
        // A yaw left over from a previous VR180 session would be silently
        // dropped by the 360 render and silently reappear on switching back.
        if (outputMode !== "vr180")
            yaw = 0
        // A photo's suggested name carries `_360_TB` or `_180x180_3dh`, and
        // those tokens are not decoration -- the Quest gallery reads the
        // filename to decide the layout, so a mode switch after the name was
        // proposed would ship a file that lies about itself. Only a name this
        // window proposed is revised; a hand-picked one is left alone, the
        // cost there being a wrong token rather than a failed render.
        if (inputPath !== "" && outputPath === suggestedOutput)
            adoptSuggestedOutput(inputPath)
    }

    Connections {
        target: app
        function onSourceInfoChanged() {
            // Before the encoder probe, which is asked about a specific
            // output size and would otherwise be run twice.
            win.adoptDefaultResolution()
            win.adoptPhotoDefaults()
            win.refreshEncoders()
            // Set the spatial-audio switch from what the file turned out to
            // be, rather than making someone notice a channel count and tick
            // a box. Forgetting it is not a small mistake: with a yaw it
            // leaves every sound at the wrong bearing, and there is nothing
            // to hear that says so.
            //
            // Only on a *new source*, so it never argues with a decision
            // already made -- changing the preview frame does not re-tick a
            // box that was deliberately cleared.
            //
            // The CLI keeps requiring the flag. Guessing on someone's behalf
            // is defensible in front of a switch they can see; it is not
            // defensible in a batch run nobody is watching.
            win.spatialAudio = win.sourceIsAmbisonic
        }
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
    // The file's own SA3D box, which is the authoritative answer and the one
    // VLC uses -- it reports "Channels: Ambisonics" for a track ffprobe
    // describes as plain 4.0, because ffprobe does not surface SA3D at all.
    readonly property bool sourceDeclaresAmbix:
        app.sourceInfo ? app.sourceInfo.declares_ambix === true : false

    // The fallback when the file says nothing: 4, 9 or 16 channels is what
    // ambiX looks like. A guess, and it cannot tell a soundfield from four
    // separate microphones -- but plenty of ambiX is delivered untagged, and
    // the cost of missing it is every sound at the wrong bearing with nothing
    // to hear that says so.
    readonly property bool sourceCountLooksAmbisonic: {
        var n = app.sourceInfo ? app.sourceInfo.audio_channels : 0
        return n === 4 || n === 9 || n === 16
    }

    readonly property bool sourceIsAmbisonic:
        sourceDeclaresAmbix || sourceCountLooksAmbisonic
    // Four cases, and nested ternaries had stopped being readable at three.
    readonly property string spatialAudioHint: {
        var turning = outputMode === "vr180" && yaw !== 0
        var channels = app.sourceInfo && app.sourceInfo.audio_channels
                       ? app.sourceInfo.audio_channels : 0

        if (spatialAudio && turning)
            return "The soundfield will be turned " + yaw.toFixed(0)
                 + "° to match the view, so sounds stay where you see them. "
                 + "The audio is re-encoded once to do it — the log names the "
                 + "codec — and the picture is unaffected."
        if (spatialAudio && sourceDeclaresAmbix)
            // No hedging needed here: the file says so itself.
            return "Set from the file, which declares its audio as ambiX in "
                 + "its own metadata — the same thing VLC reads when it says "
                 + "\"Channels: Ambisonics\"."
        if (spatialAudio && sourceCountLooksAmbisonic)
            // Here it is a guess, and it says so: a channel count cannot tell
            // ambiX from four separate microphones or a four-stem mix, and
            // treating those as a soundfield would be worse than leaving them
            // alone.
            return "Set from the file: it has " + channels + " audio "
                 + "channels, which is what ambiX looks like. It does not say "
                 + "so outright, though — untick this if those are really "
                 + "separate microphones or stems."
        if (turning)
            return "Tick this if the source audio really is ambiX: 4, 9 or 16 "
                 + "channels. Without it the view turns and the sound does "
                 + "not, leaving every source " + Math.abs(yaw).toFixed(0)
                 + "° out of place."
        return "Tick only if the source audio really is ambiX: 4, 9 or 16 "
             + "channels."
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
        if (name === "")
            return photoMode
                ? "Depth Pro: the sharpest thin structures measured. Downloads 3.6 GB on first use."
                : "Depth Anything V3: the flattest walls and floors measured, and fast enough on a CPU. Downloads 105 MB on first use."
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

    // The recommended entry first and unlabelled by model name, because which
    // model it means depends on the job. The rest keep their CLI names: the
    // dropdown is also how someone works out what to pass on the command line.
    readonly property var backendChoices: [
        {key: "", text: "Best for this job (recommended)"},
        {key: "depth-anything-v3", text: "depth-anything-v3"},
        {key: "depth-pro", text: "depth-pro"},
        {key: "auto", text: "auto — fastest runtime here"},
        {key: "depth-anything", text: "depth-anything"},
        {key: "video-depth-anything", text: "video-depth-anything"},
        {key: "onnx", text: "onnx"},
    ]
    function backendIndex(key) {
        for (var i = 0; i < backendChoices.length; ++i)
            if (backendChoices[i].key === key)
                return i
        return 0
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
        : depthBackend === "depth-anything-v3"
        ? [{key: "small", label: "Small — the default, and the best measured"},
           {key: "base",  label: "Base — flatter floors, four times the size"},
           {key: "large", label: "Large — worse on chair gaps; here for completeness"}]
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
        // V3's best variant is Small, not Base. Without this, selecting the V3
        // backend while the shared Base selection is in force would silently
        // fetch 413 MB and render the variant that measured worse.
        if (depthBackend === "depth-anything-v3" && depthModel === "base")
            depthModel = "small"
    }
    onDepthBackendChanged: _clampModel()
    onDepthModelChanged: _clampModel()

    // ---- dialogs --------------------------------------------------------
    FileDialog {
        id: openDialog
        title: "Choose a 360° video or photo"
        // From the accepted lists, never written out here: a hand-kept copy
        // is how this dialog came to hide every photo the tool could open.
        nameFilters: app.openFilters
        onAccepted: {
            win.inputPath = app.toLocalPath(selectedFile.toString())
            win.adoptSuggestedOutput(selectedFile.toString())
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
        title: win.photoMode ? "Save stereoscopic photo as"
                             : "Save stereoscopic video as"
        fileMode: FileDialog.SaveFile
        defaultSuffix: win.photoMode ? "jpg" : "mp4"
        nameFilters: app.saveFilters(win.photoMode)
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
                                    placeholderText: "Choose a video or photo…"
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
                                visible: !win.photoMode
                                hint: win.spatialAudioHint
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
                                    objectName: "resolutionBox"
                                    Layout.fillWidth: true
                                    textRole: "text"
                                    valueRole: "width"
                                    model: win.resolutionModel
                                    currentIndex: win.resolutionIndex

                                    // Everything that touches currentIndex
                                    // restores a *binding*, never a value.
                                    //
                                    // ComboBox severs the declared binding as
                                    // soon as anything writes to currentIndex
                                    // -- including a re-sync handler. Once
                                    // severed it tracks only whatever signal
                                    // that handler listened for, and goes
                                    // stale on every other input: a mode
                                    // switch, a new file, or simply the source
                                    // probe arriving after the fact.
                                    //
                                    // That is not hypothetical. Picking a
                                    // reduced size and then switching format
                                    // left the box reading "full size" while
                                    // the render used the reduced one -- a
                                    // control showing one thing and doing
                                    // another, which is the worst way for this
                                    // to fail because there is nothing to see.
                                    function trackIndex() {
                                        currentIndex = Qt.binding(
                                            function () { return win.resolutionIndex })
                                    }
                                    onActivated: {
                                        win.outputWidth =
                                            (currentValue === win.sourceWidth
                                             ? 0 : currentValue)
                                        trackIndex()
                                    }
                                    // Replacing the model resets currentIndex
                                    // to 0 without changing resolutionIndex,
                                    // so a binding alone would not re-fire.
                                    onModelChanged: trackIndex()

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
                            visible: !win.photoMode

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
                                label: "Face angular correction"
                                hint: "Straightens walls and floors that bow toward you. Depth Anything V3 only; 0 is off, 0.55-0.7 measured best. Costs about a fifth of the depth range, so pair it with a higher strength."
                                Slider {
                                    Layout.fillWidth: true
                                    from: 0; to: 1; stepSize: 0.05
                                    value: win.faceAngularCorrection
                                    onMoved: win.faceAngularCorrection = value
                                }
                                Text {
                                    text: win.faceAngularCorrection.toFixed(2)
                                    color: Theme.text
                                    font.pixelSize: Theme.fontM
                                    Layout.preferredWidth: 32
                                }
                            }

                            Row2 {
                                label: "Sharp eye"
                                hint: "Which eye keeps the original frame. The warped eye hides what is behind an occluder on one side and has to invent it on the other, so which to pick depends on the scene."
                                ComboBox {
                                    Layout.fillWidth: true
                                    model: ["Left", "Right"]
                                    currentIndex: win.sourceRight ? 1 : 0
                                    onActivated: win.sourceRight = (currentIndex === 1)
                                }
                            }

                            Row2 {
                                label: "Baseline shared"
                                hint: "How much of the separation the sharp eye also carries. 0 keeps it pristine and puts everything in the other eye; 0.5 splits evenly. The 3D effect is the same either way - this chooses where the errors land. Sharing gives far fewer holes per eye and makes the two eyes agree, at the cost of no longer having one untouched eye."
                                Slider {
                                    Layout.fillWidth: true
                                    from: 0; to: 0.5; stepSize: 0.05
                                    value: win.baselineShare
                                    onMoved: win.baselineShare = value
                                }
                                Text {
                                    text: win.baselineShare.toFixed(2)
                                    color: Theme.text
                                    font.pixelSize: Theme.fontM
                                    Layout.preferredWidth: 32
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
                                    textRole: "text"
                                    valueRole: "key"
                                    model: win.backendChoices
                                    currentIndex: win.backendIndex(
                                        win.depthBackend)
                                    onActivated: {
                                        if (win.backendUsable(currentValue))
                                            win.depthBackend = currentValue
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
                                        enabled: win.backendUsable(modelData.key)
                                        highlighted:
                                            backendBox.highlightedIndex === index
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
                                                visible: !enabled
                                                text: win.backendDetail(
                                                    modelData.key)
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
                                // Hidden for the backends that have no variant
                                // worth choosing: ONNX takes a graph path
                                // instead, Depth Pro ships one checkpoint, and
                                // the recommended entry picks its own.
                                visible: win.depthBackend !== "onnx"
                                         && win.depthBackend !== "depth-pro"
                                         && win.depthBackend !== ""
                                hint: win.depthBackend === "depth-anything-v3"
                                      ? (win.depthModel === "small"
                                         ? "The default, and capacity does not help here: Large measured worse on the chair gaps for 13× the download. 105 MB."
                                         : win.depthModel === "base"
                                           ? "Marginally flatter floors than Small, four times the download, and no better anywhere else. 413 MB."
                                           : "Measured worse than Small on the chair gaps. Offered for completeness. 1.4 GB.")
                                      : win.depthModel === "base"
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
                            visible: !win.photoMode

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
                                visible: !win.photoMode
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
                                visible: !win.photoMode
                                enabled: win.depthBackend === "video-depth-anything"
                                SpinBox {
                                    from: 0; to: 16
                                    value: win.chunkOverlap
                                    onValueModified: win.chunkOverlap = value
                                }
                            }

                            Row2 {
                                label: "Temporal fill"
                                visible: !win.photoMode
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

                    // For a photo this panel is not a preview at all. A
                    // preview exists so you can judge one frame before
                    // committing to an hour; here that frame *is* the
                    // deliverable, so the panel shows the source you opened
                    // and then the result that replaced it.
                    readonly property bool showingSource:
                        win.photoMode && !hasPreview
                                      && app.thumbnailSource !== ""

                    Image {
                        id: previewImage
                        anchors.fill: parent
                        anchors.margins: 1
                        source: previewPanel.hasPreview
                                ? app.previewSource
                                : (previewPanel.showingSource
                                   ? app.thumbnailSource : "")
                        fillMode: Image.PreserveAspectFit
                        asynchronous: true
                        cache: false

                        // Decode small. A finished photo is the full-size
                        // deliverable, and Qt refuses to decode one: an
                        // 11904x11904 stereo JPEG needs 567 MB as ARGB32 and
                        // QImageReader's allocation limit is 256 MB, so the
                        // load fails and the panel silently stays empty --
                        // after a conversion that worked and wrote the file.
                        // sourceSize is a maximum, not a size, so the small
                        // preview and thumbnail are unaffected: Qt never
                        // scales a non-scalable image *up* to reach it.
                        sourceSize.width: 2048

                        // A failed decode used to be invisible. It cannot be
                        // now -- an empty panel after a successful render is
                        // the most confusing thing this window can do.
                        onStatusChanged: if (status === Image.Error)
                            app.reportPreviewFailure(source)

                        visible: previewPanel.hasPreview
                                 || previewPanel.showingSource
                    }

                    // Which of the two you are looking at. Without this the
                    // panel changes picture on Convert with nothing saying
                    // whether that is the input or the output.
                    Rectangle {
                        visible: win.photoMode && previewImage.visible
                        anchors.top: parent.top
                        anchors.right: parent.right
                        anchors.margins: 10
                        width: stageTag.implicitWidth + 16
                        height: 24
                        radius: 5
                        color: "#c0000000"
                        Text {
                            id: stageTag
                            anchors.centerIn: parent
                            text: previewPanel.hasPreview
                                  ? "Result" : "Source photo"
                            color: previewPanel.hasPreview ? Theme.success
                                                           : Theme.textDim
                            font.pixelSize: Theme.fontS
                        }
                    }

                    ColumnLayout {
                        anchors.centerIn: parent
                        spacing: 6
                        visible: !previewPanel.hasPreview
                                 && !previewPanel.showingSource
                        Text {
                            text: win.photoMode ? "Opening the photo…"
                                                : "No preview yet"
                            color: Theme.textDim
                            font.pixelSize: Theme.fontL
                            Layout.alignment: Qt.AlignHCenter
                        }
                        Text {
                            text: win.photoMode
                                  ? "The converted photo appears here."
                                  : "Render one frame to judge strength and depth\n"
                                    + "before committing to the whole video."
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
                        // Only over the finished stereo pair. The source
                        // photo is one picture, and labelling half of it
                        // "Right eye" would be a plain lie.
                        model: previewPanel.hasPreview
                               ? ["Left eye", "Right eye"] : []
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
                    // A photo has one frame and the conversion produces the
                    // finished thing, so there is nothing here to choose and
                    // nothing to preview.
                    visible: !win.photoMode

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
