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
    property real poleCompensation: 1.0
    property bool livePreview: false
    property real livePreviewEvery: 2.0
    // Two controls the user thinks in -- which eye stays sharp, and how
    // much of the separation it carries -- and the single number the
    // CLI takes. 0 leaves the left eye untouched, 1 the right, 0.5 is
    // an even split.
    property bool sharedDetail: true
    property bool sourceRight: false
    property real baselineShare: 0.5
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

    // The upscale pre-pass. Off unless asked for
    // On is the existing behaviour and the better picture: render at the
    // source size and resize the finished eyes. Off renders at the delivered
    // size instead, which is faster and coarser -- see the hint on the row.
    property bool supersample: true
    // With a pre-pass in front, this decides how far past the delivered size
    // to upscale, so the scale has to be recomputed when it moves.
    onSupersampleChanged: adoptUpscaleScale()
    property bool upscale: false
    onUpscaleChanged: adoptDefaultResolution()
    property string upscaleModel: "amq"
    // Whether the model above was picked from the box rather than chosen for
    // the machine. A default is only right for the kind of job it was chosen
    // for, so it has to be revisited when that kind changes; a deliberate
    // pick has to survive.
    property bool upscaleModelChosen: false
    // Derived from the chosen resolution, not typed in -- see
    // `adoptUpscaleScale`. It stays a plain property because the command line
    // and the tests both set it directly, and because the pre-pass is what
    // consumes it.
    property real upscaleScale: 2.0
    // The second stage. A model with a fixed factor rarely lands on the one
    // that was asked for, and what covers the difference used to be chosen
    // silently -- so a "2x model" at 4x was quietly handing half the job to
    // spline36 without saying so. Now it is a control, and "" means the
    // model covered the whole factor by itself.
    property string upscaleResampler: ""
    property bool upscaleResamplerChosen: false
    property bool interpolate: false
    property string interpolateModel: "chr"
    property real interpolateFps: 0

    // What the core's probe said about Topaz. A plain property holding the
    // binding, so the selftest can put the window into each of the states --
    // absent, signed out, 8K source -- on a machine that has no Topaz at all.
    property var topaz: app.upscalers

    // Every answer comes from the probe rather than from a rule repeated
    // here: `offered` is its judgement that this source is below 8K, and
    // `interpolate_offered` that it is at 30 fps or below and there is at
    // least one interpolator on this machine to do it with.
    readonly property bool topazReady: topaz.available === true
                                       && topaz.offered === true
    readonly property bool topazNeedsLogin: topaz.needs_login === true
    readonly property bool interpolateReady: topaz.interpolate_offered === true
    // Upscaling is no longer Topaz's alone: a photo model does stills, and a
    // machine with only that should still be offered the control -- for a
    // photo. `offered` is the probe's width judgement and applies to both.
    // `a && a.b === true` looks like a boolean and is not: when `a` is
    // undefined -- which it is until the probe answers -- JavaScript's `&&`
    // hands back that undefined rather than false, and QML will not put it in
    // a bool. Compare first so both sides are always a boolean.
    readonly property bool photoModelReady: topaz.photo_model !== undefined
                                        && topaz.photo_model.available === true
    // The shader takes video as well as stills, unlike the photo model, so it
    // makes the control worth showing whatever the job is.
    readonly property bool shaderReady: topaz.shader !== undefined
                                        && topaz.shader.available === true
    readonly property bool upscaleReady: topaz.offered === true
                                         && (topaz.available === true
                                             || shaderReady
                                             || (photoModelReady && photoMode))
    readonly property bool canUpscale: upscaleReady
                                       && defaultUpscaler() !== ""
    readonly property bool canInterpolate: interpolateReady
                                           && defaultInterpolator() !== ""
    // The card exists for either half. Upscaling is Topaz's alone, but RIFE
    // interpolates on any machine, so a source can be worth offering one and
    // not the other.
    readonly property bool enhanceReady: upscaleReady || interpolateReady

    // What is missing but obtainable, and how much it costs to get. The card
    // used to be hidden whenever nothing was installed, which is the worst of
    // the three possible answers: "you cannot have this" and "you do not have
    // this yet" look identical, and the probe knew the difference all along.
    readonly property var fetchable: topaz.fetchable !== undefined
                                     ? topaz.fetchable : []

    // Only what this source could actually use. The two halves are judged
    // separately because they answer different questions: `offered` is the
    // width test, and an 8K source is past the point where upscaling helps;
    // `fps_offered` is the frame-rate test, and it has no opinion about width.
    // Gating both on `offered` hid the whole card for an 8K 30 fps source --
    // which cannot be upscaled and can very much be interpolated.
    readonly property var fetchNeeded: {
        var out = []
        var up = topaz.offered === true
        var fi = topaz.fps_offered === true
        for (var i = 0; i < fetchable.length; ++i) {
            var f = fetchable[i]
            if (f.kind === "interpolate" ? fi : up)
                out.push(f)
        }
        return out
    }
    readonly property bool canFetch: fetchNeeded.length > 0

    // Downloaded is not the same as usable, and the difference is invisible.
    // A source wide enough to gain from upscaling, with every model fetched
    // and none of them able to run, showed no Upscale row and said nothing --
    // so the panel looked broken rather than answered. The probe knows why in
    // every case; this is only a matter of printing it.
    readonly property string upscaleBlockedWhy: {
        if (topaz.offered !== true || upscaleReady)
            return ""
        var bits = []
        if (topaz.shader !== undefined && topaz.shader.available !== true
            && topaz.shader.reason)
            bits.push(topaz.shader.reason)
        if (photoModelReady && !photoMode)
            bits.push("The photo upscaler is for stills only, and this is a video.")
        else if (topaz.photo_model !== undefined && topaz.photo_model.available !== true
                 && topaz.photo_model.reason)
            bits.push(topaz.photo_model.reason)
        return bits.join("  ")
    }
    readonly property string fetchSummary: {
        var names = []
        for (var i = 0; i < fetchNeeded.length; ++i)
            names.push(fetchNeeded[i].label)
        return names.join(", ")
    }
    readonly property real fetchMb: {
        var mb = 0
        for (var i = 0; i < fetchNeeded.length; ++i)
            mb += fetchNeeded[i].mb
        return Math.round(mb * 10) / 10
    }
    // Fetch only what was offered, so an 8K source does not quietly pull two
    // upscalers it was just told it has no use for.
    readonly property string fetchKeys: {
        var keys = []
        for (var i = 0; i < fetchNeeded.length; ++i)
            keys.push(fetchNeeded[i].key)
        return keys.join(",")
    }

    // 0 means "just double it", which either interpolator does on its own --
    // the rate a headset wants depends on the source, and doubling 30 lands
    // at 60 either way.
    readonly property var fpsChoices: [
        { text: "Double the source rate", key: 0 },
        { text: "48 frames a second", key: 48 },
        { text: "60 frames a second", key: 60 },
        { text: "72 frames a second", key: 72 },
        { text: "90 frames a second", key: 90 },
        { text: "120 frames a second", key: 120 }
    ]

    property bool logExpanded: true

    function currentOptions() {
        return {
            "input": inputPath, "output": outputPath, "quality": quality,
            "codec": codec, "outputMode": outputMode, "yaw": yaw,
            "outputWidth": outputWidth, "sourceWidth": sourceWidth,
            "supersample": supersample,
            "strength": strength, "gradientLimit": gradientLimit,
            "faceAngularCorrection": faceAngularCorrection,
            "poleCompensation": poleCompensation,
            "livePreview": livePreview,
            "livePreviewEvery": livePreviewEvery,
            "leftShare": leftShare, "sharedDetail": sharedDetail,
            "spatialAudio": spatialAudio,
            "sourceSubsampling": sourceSubsampling,
            "faceSizeAuto": faceSizeAuto, "faceSize": faceSize,
            "depthTiles": depthTiles, "depthBackend": depthBackend,
            "depthModel": depthModel, "onnxModel": onnxModel,
            "device": device, "fgErode": fgErode, "smooth": smooth,
            "inpaint": inpaint, "chunkSize": chunkSize,
            "chunkOverlap": chunkOverlap, "temporalFill": temporalFill,
            "startFrame": startFrame, "maxFrames": maxFrames,
            "upscale": upscale && canUpscale && upscaleSelectionUsable(),
            // "" in the first dropdown is a choice, not an absence: the
            // resampler then does the whole factor, which is what won every
            // rung where the source was already sharpened. So the effective
            // upscaler is the model where there is one and the resampler
            // where there is not -- both are rows in the same table, so the
            // core takes either by code.
            "upscaleModel": upscaleModel !== "" ? upscaleModel
                                                : upscaleResampler,
            "upscaleResampler": upscaleModel !== "" ? upscaleResampler : "",
            "upscaleScale": upscaleScale,
            "interpolate": interpolate && canInterpolate
                           && interpolatorUsable(interpolateModel),
            "interpolateModel": interpolateModel,
            "interpolateFps": interpolateFps
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

    Component.onCompleted: {
        app.probeBackends()
        refreshNvvsr()
    }

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
        // Upscaling inverts which of these is the question. Off, the source
        // decides what can be delivered and this suggests a size under it.
        // On, the *delivered size* is the question and how far to upscale is
        // the answer, so the list runs above the source instead.
        if (upscale) {
            // Asked of `app` rather than read from the `resolutions` binding,
            // for the reason given at the top of this function and ignored
            // once already: this runs from `onUpscaleChanged`, and the
            // binding has not re-evaluated for the new value yet. Reading it
            // got the *downward* list, chose 3840 from it, and left a size
            // that is not in the upward list at all -- so the box fell back
            // to its first entry and showed 7680 while the readout said
            // "1.07x, 3840 to 4096". Two wrong numbers, one stale read.
            var l = app.resolutionChoices(app.sourceInfo.width,
                                          app.sourceInfo.height,
                                          outputMode, true)
            if (l.length > 0) {
                // The largest that actually decodes, not the smallest. A 4K
                // source upscaled can reach 7680x7680, which no HEVC or H.264
                // level plays -- black on a Quest 3 -- so the default is the
                // biggest entry under that ceiling and 5760 is what a 4K
                // source lands on. Picking the smallest instead would ask
                // least of the model and hand back the least resolution,
                // which is not what someone switching upscaling on wants.
                var pick = preferredWidth(l)
                if (outputWidth === 0 || outputWidth === suggestedOutputWidth
                        || outputWidth < app.sourceInfo.width)
                    outputWidth = pick
                suggestedOutputWidth = pick
            }
            adoptUpscaleScale()
            return
        }
        var w = app.defaultOutputWidth(app.sourceInfo.width,
                                       app.sourceInfo.height,
                                       outputMode,
                                       app.isImage(inputPath))
        if (outputWidth === 0 || outputWidth === suggestedOutputWidth)
            outputWidth = w
        suggestedOutputWidth = w
    }

    // How far the pre-pass has to enlarge the source to land on the size that
    // was asked for. Derived rather than chosen: a multiplier and a delivered
    // size in the same panel ask the same question in two units, and nothing
    // stops the two answers disagreeing.
    //
    // Supersample is what claims the headroom. It already means "render above
    // the delivery size and resize down", so with a pre-pass in front it
    // reaches one standard size further and lets the downsample do the rest --
    // which on the matched pair measured came out 1.5 dB ahead of upscaling
    // straight to the delivered size. Off, the pre-pass lands exactly on the
    // delivery and nothing is computed to be thrown away.
    function upscaleTargetWidth() {
        var want = outputWidth > 0 ? outputWidth : sourceWidth
        if (!supersample || sourceWidth <= 0)
            return want
        var ceiling = sourceWidth * 4          // what --upscale-scale accepts
        var best = want
        var l = app.resolutionChoices(sourceWidth,
                                      app.sourceInfo ? app.sourceInfo.height : 0,
                                      outputMode, true)
        for (var i = 0; i < l.length; ++i)
            if (l[i].width > want && l[i].width <= ceiling
                    && (best === want || l[i].width < best))
                best = l[i].width
        return best
    }

    function adoptUpscaleScale() {
        if (!upscale || sourceWidth <= 0)
            return
        var target = upscaleTargetWidth()
        if (target > 0)
            upscaleScale = target / sourceWidth
    }

    // Keep the second stage consistent with the first.
    //
    // Called whenever the model or the factor moves, because the residual is
    // computed from both: picking a 4x model after asking for 4x should empty
    // this control, and dragging the factor to 4x after picking a 2x model
    // should fill it. A hand-picked resampler survives, the same rule the
    // resolution box follows above -- but only while it still fits, since a
    // change of direction makes the old choice meaningless rather than merely
    // unfashionable.
    function adoptDefaultResampler() {
        var want = defaultResampler()
        if (!upscaleResamplerChosen) {
            upscaleResampler = want
            return
        }
        // A deliberate pick is kept unless it is no longer on offer: an
        // upscaling filter cannot finish a downscale, and vice versa.
        var l = resamplerModels()
        if (want === "" || topazIndex(l, upscaleResampler) < 0) {
            upscaleResampler = want
            upscaleResamplerChosen = false
        }
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

    // Asked per source, because whether upscaling is worth offering depends
    // on how wide this one already is -- and because a sign-in that lapsed
    // since the last file should be reported before the render, not after.
    function refreshUpscalers() {
        // Not for the empty state between two files: probing width 0 would
        // answer "worth offering", and the card would flash into view before
        // the real width arrived to say otherwise.
        if (sourceWidth > 0)
            app.probeUpscalers(sourceWidth, app.sourceInfo.fps || 0)
    }

    // The line under a model's name. Says what it does about detail that is
    // not in the source, and what factor it produces -- the two things the
    // arithmetic below depends on, so they belong where the choice is made.
    function upscalerNote(entry, usable) {
        if (entry.source === "topaz")
            return usable ? "Topaz" : "Topaz — signed out"
        var bits = []
        if (entry.category === "predictor")
            bits.push("Predictor")
        else if (entry.category === "generator")
            bits.push("Generator")
        else if (entry.category === "resampler")
            bits.push("Resampler")
        if (entry.scale === 0)
            bits.push("any factor")
        else if (entry.scale > 0)
            bits.push(entry.scale + "×")
        if (entry.short === win.fastestIn(entry.category))
            bits.push("fastest")
        return bits.join(" · ")
    }

    // The cheapest usable entry of a category. Relative cost travels between
    // machines where a millisecond count does not -- architecture fixes the
    // ordering and hardware only scales it -- so this marker is safe to show
    // without measuring anything here.
    function fastestIn(category) {
        var l = topaz.models
        if (!l || !category)
            return ""
        var best = "", bestCost = -1
        for (var i = 0; i < l.length; ++i) {
            if (l[i].category !== category || !upscalerUsable(l[i].short))
                continue
            var c = l[i].cost === undefined ? 1.0 : l[i].cost
            if (bestCost < 0 || c < bestCost) {
                bestCost = c
                best = l[i].short
            }
        }
        return best
    }

    // Everything that can take the first stage: the models, never the
    // resamplers, which are the second dropdown's business.
    function firstStageModels() {
        var l = topaz.models, out = []
        if (!l)
            return out
        for (var i = 0; i < l.length; ++i)
            if (l[i].category !== "resampler")
                out.push(l[i])
        return out
    }

    // The second dropdown's contents depend on which way the residual goes.
    // Coming back down is a different filter set -- an area filter averages
    // what it discards, where a sharpening upscaler run backwards aliases --
    // so the two are never offered together.
    function resamplerModels() {
        var l = topaz.models, out = []
        if (!l)
            return out
        var down = residualFactor() < 0.999
        for (var i = 0; i < l.length; ++i) {
            if (l[i].category !== "resampler")
                continue
            var d = l[i].direction === undefined ? "up" : l[i].direction
            if ((d === "down") === down)
                out.push(l[i])
        }
        return out
    }

    // What is left for the second stage once the first has done its part.
    // 1 means nothing is; below 1 means the model overshot and the remainder
    // is a downscale, which is a different filter set.
    function residualFactor() {
        var want = upscaleScale
        if (want <= 0)
            return 1
        if (upscaleModel === "")
            return want                     // no model: the resampler does all
        var l = topaz.models
        var i = topazIndex(l, upscaleModel)
        if (i < 0)
            return 1
        var native = l[i].scale
        if (native === undefined || native === 0)
            return 1                        // covers whatever it is asked for
        return want / native
    }

    function residualNote() {
        var r = residualFactor()
        if (Math.abs(r - 1) < 0.001)
            return ""
        if (upscaleModel === "")
            return "Doing all " + win.factorText(r) + " of it."
        if (r > 1)
            return "The model doubles; a resampler covers the remaining "
                    + win.factorText(r) + "."
        return "The model overshoots, so this scales back down by "
                + win.factorText(r) + "."
    }

    function factorText(f) {
        return (Math.round(f * 100) / 100) + "×"
    }

    // The photo model is for stills: on video it amplifies a small change
    // than the scene has, which reads as crawling. Left in the list and
    // greyed rather than hidden, so the reason can be shown.
    // Whether the pair of dropdowns names something that can actually run.
    // With no model the resampler carries it, and a resampler is a preset in
    // an ffmpeg this project already requires, so that case is always usable.
    function upscaleSelectionUsable() {
        if (upscaleModel === "")
            return upscaleResampler !== ""
        return upscalerUsable(upscaleModel)
    }

    function upscalerUsable(code) {
        var l = topaz.models
        if (!l)
            return false
        for (var i = 0; i < l.length; ++i)
            if (l[i].short === code) {
                // Nothing is barred by the kind of job any more. Three
                // models used to be refused for video on an amplification
                // figure read against the Lanczos floor -- and the floor was
                // never the target: the untouched 8K itself changed 1.20x as
                // much as Lanczos did. Watched over 120 frames, every model
                // here held still. What is left is whether the machine can
                // actually run it.
                if (l[i].source !== "topaz")
                    return true             // the free ones need no sign-in
                return topaz.needs_login !== true
            }
        return false
    }

    function defaultUpscaler() {
        var l = topaz.models
        if (!l)
            return ""
        // Artemis Medium Quality where Topaz can be used -- it is what the
        // measurements were made against -- then NVIDIA VSR where its wheel
        // is present and its licence accepted, then the best free predictor.
        //
        // The NVIDIA rung asks the probe rather than asking for a GPU: an
        // RTX card with nothing downloaded would otherwise be handed a
        // default it cannot run. `present()` on the core side answers the
        // whole question, so a model that reaches this list is one that
        // works.
        //
        // The job type no longer changes the answer. That split assumed a
        // still and a video want different models; what actually decides is
        // whether the source is degraded or pristine, which the job type
        // does not say.
        var pick = topaz.video_default
        var order = ["amq", "nvvsr_ultra"]
        if (pick)
            order.push(pick)
        for (var p = 0; p < order.length; ++p)
            for (var i = 0; i < l.length; ++i)
                if (l[i].short === order[p] && upscalerUsable(order[p]))
                    return order[p]
        for (i = 0; i < l.length; ++i)
            if (l[i].category !== "resampler"
                    && upscalerUsable(l[i].short))
                return l[i].short
        return ""
    }

    // The second stage, chosen for the residual the first one leaves.
    //
    // "" when the model covered the whole factor. Otherwise the sharpest
    // resampler on the way up, because that is the case where the choice
    // actually matters -- with no model in front of it the resampler is
    // doing all the work, and the five differed by 0.37 dB at 2x. After a
    // model they differ by 0.02 to 0.08 dB, which is nothing, so the same
    // default serves both without argument.
    function defaultResampler() {
        var r = residualFactor()
        if (Math.abs(r - 1) < 0.001)
            return ""
        // Going down is a different filter set. libplacebo takes it on
        // `downscaler` and the onnx path uses INTER_AREA; a sharpening
        // upscaler has no meaning here.
        if (r < 1)
            return "area"
        var want = ["ewa_lanczos4sharpest", "lanczos", "spline36"]
        var l = topaz.models
        if (!l)
            return ""                       // before the probe lands
        for (var w = 0; w < want.length; ++w)
            for (var i = 0; i < l.length; ++i)
                if (l[i].short === want[w])
                    return want[w]
        return ""
    }

    // Which interpolator to use unless someone picks another. Chronos when
    // Topaz is here and signed in -- it is what the measurements were made
    // against -- and RIFE otherwise, which is free and needs nothing. "" when
    // the machine has neither, which is what disables the control.
    function defaultInterpolator() {
        var l = topaz.interpolators
        if (!l)
            return ""
        for (var i = 0; i < l.length; ++i)
            if (l[i].short === "chr" && interpolatorUsable("chr"))
                return "chr"
        for (i = 0; i < l.length; ++i)
            if (interpolatorUsable(l[i].short))
                return l[i].short
        return ""
    }

    // A Topaz model cannot be used while Topaz is signed out; RIFE does not
    // care, which is the point of having it in the same list.
    function interpolatorUsable(code) {
        var l = topaz.interpolators
        if (!l)
            return false
        for (var i = 0; i < l.length; ++i)
            if (l[i].short === code)
                // Reads `topaz` rather than the `topazNeedsLogin` binding:
                // this is called from onTopazChanged, where a binding derived
                // from the same property has not necessarily caught up.
                return l[i].source !== "topaz" || topaz.needs_login !== true
        return false
    }

    // A choice that this machine cannot honour is worse than no choice: it
    // would build a command the core refuses. Only replaced when it has
    // stopped being usable, so a deliberate pick survives the next probe.
    onTopazChanged: {
        if (!interpolatorUsable(interpolateModel))
            interpolateModel = defaultInterpolator()
        // "" means two things and only one of them is a decision. Someone
        // who picked "None" wants no model and must keep it; "" also happens
        // before the probe lands, when `defaultUpscaler` has no list to
        // choose from and answers with nothing. `upscaleModelChosen` is what
        // separates them -- without it the first case swallowed the second
        // and the box sat empty on a machine with Topaz installed.
        if (!deliberatelyNoModel() && !upscalerUsable(upscaleModel))
            upscaleModel = defaultUpscaler()
        adoptDefaultResampler()
    }

    // Whether the empty first dropdown is an answer rather than an absence.
    function deliberatelyNoModel() {
        return upscaleModel === "" && upscaleModelChosen
    }

    // ---- NVIDIA VSR -------------------------------------------------------
    //
    // Held here rather than read from `app` at every use, because asking runs
    // `nvidia-smi` and reads two PDFs. Refreshed when the state can have
    // moved: at startup, after the wheel arrives, and after the licence is
    // answered.
    property var nvvsr: ({})
    function refreshNvvsr() { nvvsr = app.nvvsrStatus() }
    // Set while a download this window started is in flight, so the licence
    // is put up the moment the wheel lands rather than waiting for the first
    // render. Someone who has just spent 490 MB is the right person to ask,
    // and the terms only become readable at that point -- they are inside
    // the wheel.
    property bool nvvsrAsking: false
    Connections {
        target: app
        function onNvvsrChanged() {
            win.refreshNvvsr()
            if (win.nvvsrAsking && win.nvvsrAction === "licence") {
                win.nvvsrAsking = false
                win.showNvidiaLicence()
            } else if (win.nvvsrAction !== "licence") {
                win.nvvsrAsking = false   // it failed, or it is already ours
            }
        }
    }

    // Show the terms. Refuses rather than improvises when they cannot be
    // read: an empty string leaves the dialog's Accept disabled and says so,
    // which is the only honest thing to do with someone else's licence.
    function showNvidiaLicence() {
        nvidiaLicence.text = app.nvvsrAgreement()
        nvidiaLicence.open()
    }

    // What the row offers, which is three different things.
    readonly property string nvvsrAction: {
        if (!nvvsr.supported)
            return ""                       // not this machine; why_not says so
        if (!nvvsr.installed)
            return "download"
        if (!nvvsr.accepted)
            return "licence"
        return ""                           // ready, and in the dropdown
    }

    // A photo and a video are offered different upscalers, so the choice has
    // to be revisited when the kind of job changes as well -- and here that
    // means any choice this made itself, not only one that has stopped being
    // usable. The two defaults are both usable on both kinds of job, so
    // testing usability alone changed nothing: the interface opens with no
    // input, the probe lands while this is still a video, and the video
    // default then stayed put when a photo was opened.
    onPhotoModeChanged: {
        if (!upscaleModelChosen
                || (!deliberatelyNoModel() && !upscalerUsable(upscaleModel)))
            upscaleModel = defaultUpscaler()
        adoptDefaultResampler()
    }

    function fpsIndex(fps) {
        for (var i = 0; i < fpsChoices.length; ++i)
            if (fpsChoices[i].key === fps)
                return i
        return 0
    }

    // Which entry of a probed model list a short code sits at, for the
    // ComboBoxes below. -1 when the list has not arrived yet -- which is the
    // state before the probe answers, and on every machine without Topaz --
    // leaving the box empty rather than showing the wrong model as chosen.
    function topazIndex(list, code) {
        if (!list)
            return -1
        for (var i = 0; i < list.length; ++i)
            if (list[i].short === code)
                return i
        return -1
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

    onOutputWidthChanged: {
        refreshEncoders()
        // The delivered size is the question now; how far to upscale is the
        // answer, and it follows from here rather than being set beside it.
        adoptUpscaleScale()
    }

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
            win.refreshUpscalers()
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

    // The size the stereo pass will actually see, which is not the source's
    // when a pre-pass runs first. Every sizing decision below reads this
    // rather than `sourceWidth`: upscaling a 4K source 2x renders 7680x7680
    // in 360 mode, which no HEVC or H.264 level decodes -- the same ceiling
    // `default_output_width` exists to keep an 8K *source* under. Judged from
    // the source alone, a 4K input looks comfortably below the cap and the
    // upscale carries it over unremarked, discovered after the render.
    readonly property int effectiveWidth:
        upscale && upscaleScale > 0 ? Math.round(sourceWidth * upscaleScale)
                                    : sourceWidth
    readonly property int effectiveHeight:
        app.sourceInfo && app.sourceInfo.height
        ? (upscale && upscaleScale > 0
           ? Math.round(app.sourceInfo.height * upscaleScale)
           : app.sourceInfo.height)
        : 0

    // Built from the *source*, not from the upscaled size, when a pre-pass is
    // running: the list is what to deliver, and how far to upscale follows
    // from the choice. Reading it from `effectiveWidth` -- which is itself
    // derived from the scale -- would close a loop through the control that
    // sets the scale.
    readonly property var resolutions:
        sourceWidth > 0 && upscale
        ? app.resolutionChoices(sourceWidth,
                                app.sourceInfo ? app.sourceInfo.height : 0,
                                outputMode, true)
        : (effectiveWidth > 0
           ? app.resolutionChoices(effectiveWidth, effectiveHeight, outputMode)
           : [])

    readonly property var outputSize:
        effectiveWidth > 0
        ? app.outputSize(effectiveWidth, effectiveHeight, outputMode,
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
                // The same limit, and the same exception: a photo is
                // decoded as an image, so telling its reader that 7680 is
                // "upload only" would be talking about a codec that never
                // runs on it.
                sub: r.megapixels + " MP"
                     + (photoMode ? ""
                        : r.fits ? " — plays on a headset"
                                 : " — past the 35.6 MP decode limit; "
                                   + "upload only")
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

    // A width the list does not contain shows as entry 0 by the rule above,
    // so the box says one size while the render uses another -- and says it
    // confidently, which is worse than saying nothing. That is how 7680
    // appeared beside a readout describing 4096. Snapped back whenever the
    // list changes under it, which is what turning upscaling on does.
    onResolutionsChanged: reconcileOutputWidth()
    function reconcileOutputWidth() {
        var l = resolutions
        if (l.length === 0)
            return
        var want = outputWidth === 0 ? sourceWidth : outputWidth
        for (var i = 0; i < l.length; ++i)
            if (l[i].width === want)
                return
        // The same rule the default uses, not `l[0]`. This can fire while
        // `adoptDefaultResolution` is still mid-flight -- the list re-
        // evaluates the moment `upscale` changes, before the width has been
        // chosen -- and falling back to the first entry then picked 7680,
        // the one size a headset will not decode.
        for (var j = 0; j < l.length; ++j)
            if (l[j].width === suggestedOutputWidth) {
                outputWidth = suggestedOutputWidth
                return
            }
        outputWidth = preferredWidth(l)
    }

    // The largest entry that a decoder will actually play, or the smallest
    // if none of them will. 7680x7680 is 59 MP against the 35.7 MP ceiling,
    // so a 4K source upscaling lands on 5760 rather than on the top of the
    // list.
    //
    // A photo takes the top of the list instead, because that ceiling is the
    // *video* decoder's and a still never reaches one: a 59 MP stereo JPEG
    // displays fine on a Quest 3. `defaultOutputWidth` has always known this
    // and returns 0 for a photo; the upscaling path reached for `fits`
    // directly and so re-imposed a limit that had already been lifted,
    // capping an upscaled photo at 5760 for no reason.
    function preferredWidth(l) {
        if (!l || l.length === 0)
            return 0
        if (photoMode)
            return l[0].width              // largest first
        for (var i = 0; i < l.length; ++i)
            if (l[i].fits === true)
                return l[i].width
        return l[l.length - 1].width
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
                ? "Depth Pro: the sharpest thin structures measured. Downloads 1.9 GB on first use, and wants a GPU — on a processor it measured 9.4 GB of RAM and half an hour for one photo."
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

    // Backend and variant as one list, because separately they were a puzzle:
    // which variants existed depended on the backend, two of the backends had
    // no variant at all so the second row vanished, and the shared variant
    // selection carried across a backend change and had to be clamped back.
    // Spelling the valid pairs out removes all three problems -- every entry
    // is a thing you can actually run, and there is nothing to keep in sync.
    readonly property var depthVariants: ({
        "depth-anything-v3": [["small", "Small"], ["base", "Base"],
                              ["large", "Large"]],
        "depth-pro": [],
        "auto": [["small", "Small"], ["base", "Base"], ["large", "Large"]],
        "depth-anything": [["small", "Small"], ["base", "Base"],
                           ["large", "Large"]],
        // Ships small and large only; Base would fail at load.
        "video-depth-anything": [["small", "Small"], ["large", "Large"]],
        "onnx": []
    })

    // `depth-anything` is V2 -- it was named before there was a V3 to tell it
    // apart from, and it is a CLI value, so it cannot simply be renamed here
    // without the dropdown ceasing to say what you would type. The generation
    // goes in the label instead, alongside the flag rather than replacing it.
    function depthLabel(backend) {
        return backend === "depth-anything" ? "depth-anything  (v2)" : backend
    }

    readonly property var depthChoices: {
        var out = [{backend: "", model: "",
                    text: "Best for this job (recommended)"}]
        var order = ["depth-anything-v3", "depth-pro", "auto",
                     "depth-anything", "video-depth-anything", "onnx"]
        for (var i = 0; i < order.length; ++i) {
            var b = order[i]
            var vs = depthVariants[b]
            if (vs.length === 0) {
                out.push({backend: b, model: "", text: depthLabel(b)})
            } else {
                for (var j = 0; j < vs.length; ++j)
                    out.push({backend: b, model: vs[j][0],
                              text: depthLabel(b) + "  ·  " + vs[j][1]})
            }
        }
        return out
    }

    function depthChoiceIndex(backend, model) {
        var fallback = 0
        for (var i = 0; i < depthChoices.length; ++i) {
            var c = depthChoices[i]
            if (c.backend !== backend)
                continue
            if (c.model === "" || c.model === model)
                return i
            if (fallback === 0)
                fallback = i      // right backend, variant not in its list
        }
        return fallback
    }

    // The detail line for a pair: the backend's own text, plus what the
    // variant costs where that is the thing worth knowing.
    function depthChoiceDetail(backend, model) {
        if (backend === "depth-anything-v3")
            return model === "base"
                ? "Marginally flatter floors than Small, four times the download, and no better anywhere else. 413 MB."
                : model === "large"
                  ? "Measured worse than Small on the chair gaps. Offered for completeness. 1.4 GB."
                  : "The default, and capacity does not help here: Large measured worse on the chair gaps for 13× the download. 105 MB."
        if (backend === "depth-anything" || backend === "auto"
                || backend === "video-depth-anything")
            return model === "large"
                ? "Twice Base's extra cost and measured no better on 8K footage. Downloads ~1.3 GB on first use."
                : model === "small"
                  ? "The sharpest of the V2 family, and the noisiest — that noise is what makes thin structures shift between frames. About 9% faster overall than Base."
                  : "The best of the V2 family: lowest depth noise and 40% less flicker than Small on 8K footage. Not the tool's default any more — that is Depth Anything V3 for video and Depth Pro for stills. Downloads ~400 MB on first use."
        return backendDetail(backend)
    }

    // Checked in both directions rather than derived from the choice list: a
    // property changed handler can run before the bindings that depend on the
    // same property have re-evaluated, so reading the list here would see the
    // previous backend's. Explicit is order-independent.
    function _clampModel() {
        // Still a validity rule: the temporal backend has no Base checkpoint,
        // so a stored or defaulted Base would fail at load. The single list
        // never offers that pair, but a value can arrive from elsewhere.
        //
        // The V3 clamp that used to sit here is gone. It existed because the
        // variant selection was shared across backends and Base could carry
        // into V3 unasked; now every entry names its own variant, so choosing
        // V3 Base is a deliberate act and forcing it back to Small would just
        // ignore the user.
        if (depthBackend === "video-depth-anything" && depthModel === "base")
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

    // NVIDIA's terms, shown once before their software is first used.
    //
    // After the download rather than before it, because the agreement is
    // inside the wheel -- and NVIDIA's web page is not the same document, so
    // presenting that would ask someone to accept text other than the terms
    // they are bound by. It also matches the agreement, which binds on use
    // rather than on acquisition.
    LicenceDialog {
        id: nvidiaLicence
        vendor: "NVIDIA"
        product: "NVIDIA VSR"
        moreInfo: win.nvvsr.agreement_url || ""
        onAgreed: function (agreementText) {
            app.nvvsrAccept(agreementText)
            win.refreshNvvsr()
        }
        onRejected: win.refreshNvvsr()
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
                    // Named so a selftest screenshot can reach the sections
                    // below the fold, which is most of the panel.
                    objectName: "settingsScroll"
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

                            // Only where a smaller size was actually chosen:
                            // at full size there is nothing to render smaller
                            // than, and the switch would promise a saving it
                            // cannot make.
                            Row2 {
                                objectName: "supersampleRow"
                                label: "Supersample"
                                // Never for a photo: a still renders one
                                // frame, so trading its geometry for speed
                                // buys seconds and costs the deliverable.
                                // The CLI refuses the flag outright, so
                                // offering the switch here was offering a
                                // setting that could only fail the run.
                                visible: !photoMode
                                         && win.outputWidth !== 0
                                         && win.outputWidth !== win.sourceWidth
                                hint: win.supersample
                                      ? "Renders each eye at the source size and resizes it down, which smooths edges and keeps depth at full resolution. This is what makes a smaller output cost the same as a full-size one."
                                      : "Renders at the delivered size instead. Measured 1.65x faster on an 8K source delivered at 5760. Depth is estimated from the smaller frame, so the geometry is coarser and not only the picture — worth checking a depth edge in the headset before trusting it on a long render."
                                Switch {
                                    objectName: "supersampleSwitch"
                                    checked: win.supersample
                                    onToggled: win.supersample = checked
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

                        // ---- Topaz pre-pass ------------------------------
                        // Absent unless Topaz Video AI is installed here and
                        // this source is below 8K. Both answers come from the
                        // core's own probe, so the rule lives in one tested
                        // place rather than being restated in the window.
                        Card {
                            objectName: "topazCard"
                            title: "Enhance the source"
                            subtitle: "before the 3D pass"
                            visible: win.enhanceReady || win.canFetch
                                     || win.upscaleBlockedWhy !== ""

                            Rectangle {
                                objectName: "upscaleBlocked"
                                Layout.fillWidth: true
                                visible: win.upscaleBlockedWhy !== ""
                                implicitHeight: blockedText.implicitHeight + 16
                                color: "#2a2114"
                                radius: 6
                                border.width: 1
                                border.color: Theme.warn

                                Text {
                                    id: blockedText
                                    anchors.fill: parent
                                    anchors.margins: 8
                                    text: "Upscaling is not available for this "
                                          + "source. " + win.upscaleBlockedWhy
                                    color: Theme.warn
                                    font.pixelSize: Theme.fontS
                                    wrapMode: Text.WordWrap
                                }
                            }

                            // Ahead of everything, because with nothing
                            // installed there is nothing else in the card and
                            // this is the only thing worth saying.
                            Rectangle {
                                objectName: "enhanceFetch"
                                Layout.fillWidth: true
                                visible: win.canFetch
                                implicitHeight: fetchCol.implicitHeight + 16
                                color: "#14202a"
                                radius: 6
                                border.width: 1
                                border.color: Theme.accent

                                ColumnLayout {
                                    id: fetchCol
                                    anchors.fill: parent
                                    anchors.margins: 8
                                    spacing: 6

                                    Text {
                                        Layout.fillWidth: true
                                        text: app.fetchingEnhancers
                                              ? "Downloading " + win.fetchSummary + "..."
                                              : "Not downloaded yet: " + win.fetchSummary
                                                + ". About " + win.fetchMb
                                                + " MB in total."
                                        color: Theme.text
                                        font.pixelSize: Theme.fontS
                                        wrapMode: Text.WordWrap
                                    }

                                    Button {
                                        objectName: "enhanceFetchButton"
                                        text: app.fetchingEnhancers
                                              ? "Downloading..." : "Download"
                                        enabled: !app.fetchingEnhancers
                                                 && !app.running
                                        onClicked: app.fetchEnhancers(win.fetchKeys)
                                    }
                                }
                            }

                            // First thing in the card, because a signed-out
                            // Topaz does not refuse the job -- it renders it
                            // watermarked, which is only discovered at the end
                            // of an hour of work.
                            Rectangle {
                                objectName: "topazLogin"
                                Layout.fillWidth: true
                                visible: win.topazNeedsLogin
                                implicitHeight: loginText.implicitHeight + 16
                                color: "#2a2114"
                                radius: 6
                                border.width: 1
                                border.color: Theme.warn

                                Text {
                                    id: loginText
                                    anchors.fill: parent
                                    anchors.margins: 8
                                    text: win.topazReady
                                          ? "Please sign in to Topaz Video AI. It is installed here but signed out, and it watermarks anything it renders until you open it and sign in. Choose the file again once you have."
                                          : "Please sign in to Topaz Video AI to use its models. It is installed here but signed out. Choose the file again once you have."
                                    color: Theme.warn
                                    font.pixelSize: Theme.fontS
                                    wrapMode: Text.WordWrap
                                }
                            }

                            Row2 {
                                label: "Upscale"
                                visible: win.upscaleReady
                                hint: win.upscale
                                      ? "Runs before the stereo pass, never per eye - these models invent detail, and inventing different detail for each eye would hand the viewer rivalry instead of sharpness. Expect it to roughly double how long the whole job takes."
                                      : "Off. A 4K source converts as it is. Turning this on rebuilds it near 8K first, which is what a headset can actually show."
                                Switch {
                                    enabled: win.canUpscale
                                    checked: win.upscale && win.canUpscale
                                    onToggled: win.upscale = checked
                                }
                            }

                            Row2 {
                                label: "Amount"
                                visible: win.upscale && win.upscaleReady
                                // Shown rather than asked. It used to be a
                                // slider beside the resolution picker, which
                                // put the same question in the panel twice in
                                // two different units -- and let the two
                                // answers disagree: 2x on a 3840 source
                                // delivered at 5760 computed 7680 pixels and
                                // threw a quarter of them away.
                                hint: {
                                    if (win.sourceWidth <= 0)
                                        return "Set by the resolution above."
                                    var to = Math.round(win.sourceWidth
                                                        * win.upscaleScale)
                                    var said = win.sourceWidth + " to " + to
                                               + " wide, set by the resolution above."
                                    if (win.supersample && to > win.outputWidth
                                            && win.outputWidth > 0)
                                        said += " Past the delivered "
                                                + win.outputWidth
                                                + " because supersampling asks"
                                                + " for the headroom; it is"
                                                + " resized down afterwards."
                                    return said
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: win.upscaleScale.toFixed(2) + "×"
                                    color: Theme.text
                                    font.pixelSize: Theme.fontM
                                }
                            }

                            Row2 {
                                label: "Upscale model"
                                visible: win.upscale && win.upscaleReady
                                hint: {
                                    if (win.upscaleModel === "")
                                        return "No model. The resampler below does the whole factor, which is the safest choice on footage that is already sharpened or compressed."
                                    var l = win.topaz.models
                                    var i = win.topazIndex(l, win.upscaleModel)
                                    if (i < 0)
                                        return ""
                                    if (l[i].source === "topaz" && win.topazNeedsLogin)
                                        return "Needs a signed-in Topaz."
                                    return l[i].desc
                                }
                                ComboBox {
                                    id: upscaleBox
                                    objectName: "upscaleBox"
                                    Layout.fillWidth: true
                                    enabled: win.canUpscale
                                    textRole: "name"
                                    valueRole: "short"
                                    // Models only. The resamplers are the
                                    // second dropdown's business, and "None"
                                    // leads because choosing no model at all
                                    // is a real answer here rather than an
                                    // absence -- it is what won every rung
                                    // where the source was already sharpened.
                                    model: [{"short": "", "name": "None",
                                             "desc": "", "category": "",
                                             "scale": -1}].concat(
                                                win.firstStageModels())
                                    // A ComboBox given a model where it had
                                    // none picks its own currentIndex -- 0,
                                    // the first entry -- and that assignment
                                    // replaces any binding on it. The list
                                    // arrives from the probe, always after
                                    // this is built, so a binding here was
                                    // destroyed every single time: the box
                                    // sat on the first upscaler while the
                                    // render used the chosen one, which is
                                    // the one failure this panel cannot
                                    // afford. So own the index instead, and
                                    // re-assert it from the two things it
                                    // depends on. Deferred where the control
                                    // is mid-assignment, since it writes
                                    // after the signal that says it changed.
                                    function sync() {
                                        currentIndex = win.topazIndex(
                                            model, win.upscaleModel)
                                    }
                                    Component.onCompleted: sync()
                                    onCountChanged: Qt.callLater(sync)
                                    Connections {
                                        target: win
                                        function onUpscaleModelChanged() {
                                            upscaleBox.sync()
                                            win.adoptDefaultResampler()
                                        }
                                        function onUpscaleScaleChanged() {
                                            win.adoptDefaultResampler()
                                        }
                                        function onTopazChanged() {
                                            Qt.callLater(upscaleBox.sync)
                                        }
                                    }
                                    onActivated: {
                                        if (currentValue === ""
                                                || win.upscalerUsable(currentValue)) {
                                            win.upscaleModel = currentValue
                                            win.upscaleModelChosen = true
                                            win.upscaleResamplerChosen = false
                                            win.adoptDefaultResampler()
                                        } else
                                            currentIndex = win.topazIndex(
                                                model, win.upscaleModel)
                                    }

                                    delegate: ItemDelegate {
                                        width: upscaleBox.width
                                        enabled: modelData.short === ""
                                                 || win.upscalerUsable(modelData.short)
                                        highlighted:
                                            upscaleBox.highlightedIndex === index
                                        contentItem: ColumnLayout {
                                            spacing: 0
                                            Text {
                                                text: modelData.name
                                                color: enabled ? Theme.text
                                                               : Theme.textFaint
                                                font.pixelSize: Theme.fontM
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                            Text {
                                                text: modelData.short === ""
                                                      ? "Resampler only"
                                                      : win.upscalerNote(modelData,
                                                                         enabled)
                                                visible: text !== ""
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

                            // NVIDIA VSR is in the registry on every
                            // machine and installed on almost none, so it
                            // needs a way in that the other models do not:
                            // 490 MB from NVIDIA, under NVIDIA's terms. The
                            // row is silent once it is ready, and absent on
                            // a machine that could not run it -- with the
                            // reason shown rather than a greyed control and
                            // nothing beside it.
                            Row2 {
                                label: "NVIDIA VSR"
                                // `why_not` is null where there is no reason,
                                // and `a || null` is null rather than false --
                                // which QML then declines to assign to a bool
                                // and says so on every startup. Compared
                                // against "" so the expression is a boolean
                                // whatever the map holds.
                                visible: win.upscale && win.upscaleReady
                                         && (win.nvvsrAction !== ""
                                             || (win.nvvsr.why_not || "") !== "")
                                hint: {
                                    if (win.nvvsr.why_not)
                                        return win.nvvsr.why_not
                                    if (app.nvvsrFetching)
                                        return "Downloading about 490 MB from NVIDIA. This takes a while."
                                    if (win.nvvsrAction === "download")
                                        return "NVIDIA's RTX Video upscaler. About 490 MB from NVIDIA, under their licence, which you are asked to read once when it arrives. It reproduced this camera's own texture more closely than anything else measured."
                                    if (win.nvvsrAction === "licence")
                                        return "Downloaded. Read and accept NVIDIA's licence to use it."
                                    return ""
                                }
                                Button {
                                    Layout.fillWidth: true
                                    visible: win.nvvsrAction !== ""
                                    enabled: !app.nvvsrFetching && !app.busy
                                    text: app.nvvsrFetching
                                          ? "Downloading..."
                                          : win.nvvsrAction === "download"
                                            ? "Download (490 MB)"
                                            : "Read the licence"
                                    onClicked: {
                                        if (win.nvvsrAction === "download") {
                                            win.nvvsrAsking = true
                                            app.nvvsrInstall()
                                        } else
                                            win.showNvidiaLicence()
                                    }
                                }
                            }

                            Row2 {
                                label: "Then resample"
                                visible: win.upscale && win.upscaleReady
                                // What the model leaves behind, said out
                                // loud. It was always happening -- a 2x
                                // model asked for 4x has always handed the
                                // rest to spline36 -- and never shown, so
                                // the arithmetic looked like a single step
                                // that it never was.
                                hint: {
                                    var note = win.residualNote()
                                    if (note !== "")
                                        return note
                                    return "Nothing left to do: the model covers the whole factor by itself."
                                }
                                ComboBox {
                                    id: resampleBox
                                    objectName: "resampleBox"
                                    Layout.fillWidth: true
                                    enabled: win.canUpscale
                                             && win.residualNote() !== ""
                                    textRole: "name"
                                    valueRole: "short"
                                    model: win.residualNote() === ""
                                           ? [{"short": "", "name": "None",
                                               "desc": "", "category": "",
                                               "scale": -1}]
                                           : win.resamplerModels()
                                    function sync() {
                                        currentIndex = win.topazIndex(
                                            model, win.upscaleResampler)
                                    }
                                    Component.onCompleted: sync()
                                    onCountChanged: Qt.callLater(sync)
                                    Connections {
                                        target: win
                                        function onUpscaleResamplerChanged() {
                                            resampleBox.sync()
                                        }
                                        function onTopazChanged() {
                                            Qt.callLater(resampleBox.sync)
                                        }
                                    }
                                    onActivated: {
                                        win.upscaleResampler = currentValue
                                        win.upscaleResamplerChosen = true
                                    }

                                    delegate: ItemDelegate {
                                        width: resampleBox.width
                                        highlighted:
                                            resampleBox.highlightedIndex === index
                                        contentItem: ColumnLayout {
                                            spacing: 0
                                            Text {
                                                text: modelData.name
                                                color: Theme.text
                                                font.pixelSize: Theme.fontM
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                            Text {
                                                text: modelData.short === ""
                                                      ? ""
                                                      : win.factorText(
                                                            win.residualFactor())
                                                        + " · "
                                                        + win.upscalerNote(modelData,
                                                                           true)
                                                visible: text !== ""
                                                color: Theme.textFaint
                                                font.pixelSize: Theme.fontS
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                        }
                                    }
                                }
                            }

                            Row2 {
                                label: "Smooth motion"
                                visible: !win.photoMode && win.interpolateReady
                                hint: win.interpolate
                                      ? "Worth more in a headset than on a monitor: 30 fps judders when your head keeps moving and there is no shutter to hide it. It also multiplies the frames the stereo pass then has to convert."
                                      : "Off. The output keeps the source frame rate."
                                Switch {
                                    enabled: win.canInterpolate
                                    checked: win.interpolate
                                             && win.canInterpolate
                                    onToggled: win.interpolate = checked
                                }
                            }

                            Row2 {
                                label: "Motion model"
                                visible: win.interpolate && !win.photoMode
                                         && win.interpolateReady
                                hint: {
                                    var l = win.topaz.interpolators
                                    var i = win.topazIndex(l, win.interpolateModel)
                                    if (i < 0)
                                        return ""
                                    return l[i].source === "topaz"
                                           && win.topazNeedsLogin
                                           ? "Needs a signed-in Topaz. RIFE does not."
                                           : l[i].desc
                                }
                                ComboBox {
                                    id: interpolateBox
                                    objectName: "interpolateBox"
                                    Layout.fillWidth: true
                                    enabled: win.canInterpolate
                                    textRole: "name"
                                    valueRole: "short"
                                    model: win.topaz.interpolators
                                    // The same as the upscaler box above, for
                                    // the same reason: this list also arrives
                                    // from the probe.
                                    function sync() {
                                        currentIndex = win.topazIndex(
                                            win.topaz.interpolators,
                                            win.interpolateModel)
                                    }
                                    Component.onCompleted: sync()
                                    onCountChanged: Qt.callLater(sync)
                                    Connections {
                                        target: win
                                        function onInterpolateModelChanged() {
                                            interpolateBox.sync()
                                        }
                                        function onTopazChanged() {
                                            Qt.callLater(interpolateBox.sync)
                                        }
                                    }
                                    // A signed-out Topaz leaves its own models
                                    // in the list but greyed, rather than
                                    // vanishing them: the reason they cannot
                                    // be used is worth showing.
                                    onActivated: {
                                        if (win.interpolatorUsable(currentValue))
                                            win.interpolateModel = currentValue
                                        else
                                            currentIndex = win.topazIndex(
                                                win.topaz.interpolators,
                                                win.interpolateModel)
                                    }

                                    delegate: ItemDelegate {
                                        width: interpolateBox.width
                                        enabled: win.interpolatorUsable(
                                            modelData.short)
                                        highlighted:
                                            interpolateBox.highlightedIndex === index
                                        contentItem: ColumnLayout {
                                            spacing: 0
                                            Text {
                                                text: modelData.name
                                                color: enabled ? Theme.text
                                                               : Theme.textFaint
                                                font.pixelSize: Theme.fontM
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                            Text {
                                                text: win.upscalerNote(modelData,
                                                                       enabled)
                                                visible: text !== ""
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

                            Row2 {
                                label: "Frame rate"
                                visible: win.interpolate && !win.photoMode
                                         && win.interpolateReady
                                hint: win.interpolateFps > 0
                                      ? "Interpolated to " + win.interpolateFps.toFixed(0) + " frames a second."
                                      : "Twice the source rate, which is what 30 fps footage wants."
                                ComboBox {
                                    id: fpsBox
                                    Layout.fillWidth: true
                                    enabled: win.canInterpolate
                                    textRole: "text"
                                    valueRole: "key"
                                    model: win.fpsChoices
                                    currentIndex: win.fpsIndex(win.interpolateFps)
                                    onActivated: win.interpolateFps = currentValue
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
                                label: "Level the ground"
                                hint: "The 360 stereo format separates the eyes along the horizontal, so it gives a point below the horizon less disparity than its distance deserves and you read it as further away. Flat ground becomes a funnel: right at the horizon, twice too deep at 60 degrees down, and collapsing underfoot -- which makes the middle distance look like a raised plateau. This cancels that. 2 holds the ground level to 60 degrees below the horizon, 3 to 70, 5 to 80; 1 is off. No depth model can fix this, because the error is in the projection rather than the depth."
                                Slider {
                                    Layout.fillWidth: true
                                    from: 1; to: 5; stepSize: 0.5
                                    value: win.poleCompensation
                                    onMoved: win.poleCompensation = value
                                }
                                Text {
                                    text: win.poleCompensation.toFixed(1)
                                    color: Theme.text
                                    font.pixelSize: Theme.fontM
                                    Layout.preferredWidth: 32
                                }
                            }

                            Row2 {
                                label: "Detail on smooth depth"
                                hint: "Warp fine detail on a smoothed depth instead of the real one, so it cannot shear where the depth map puts a boundary in the wrong place. Thin structures then arrive whole and the same way in both eyes, at the cost of sitting at a slightly wrong depth. Costs a second warp per eye."
                                Switch {
                                    checked: win.sharedDetail
                                    onToggled: win.sharedDetail = checked
                                }
                            }

                            Row2 {
                                label: "Sharp eye"
                                hint: "Which eye keeps the original frame. Only applies below a 0.5 share - at 0.5 the separation is even and neither eye is the original. The warped eye hides what is behind an occluder on one side and has to invent it on the other, so which to pick depends on the scene."
                                ComboBox {
                                    Layout.fillWidth: true
                                    model: ["Left", "Right"]
                                    currentIndex: win.sourceRight ? 1 : 0
                                    onActivated: win.sourceRight = (currentIndex === 1)
                                }
                            }

                            Row2 {
                                label: "Baseline shared"
                                hint: "How much of the separation the sharp eye also carries. 0.5, the default, splits it evenly so a depth error is spread over both eyes rather than concentrated in one; 0 keeps one eye pristine and puts everything in the other. The 3D effect is the same either way - this chooses where the errors land. An even split measured equal or better on every tracked feature and pulls further ahead at wider baselines, at the cost of no eye being the untouched original."
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
                                label: "Live preview"
                                visible: !win.photoMode
                                hint: win.livePreview
                                      ? "Shows the frame being written, at most every " + win.livePreviewEvery.toFixed(0) + "s. Measured at 8K it costs about 5 ms a frame against 1700 ms to render one, and the interval is checked once per frame — so a render slower than the interval previews every frame and never writes the same picture twice."
                                      : "Off. The preview panel stays empty until the render finishes. Turning this on shows the frame being written as it goes, for about 5 ms a frame at 8K."
                                Switch {
                                    checked: win.livePreview
                                    onToggled: win.livePreview = checked
                                }
                            }

                            Row2 {
                                label: "Preview every"
                                visible: !win.photoMode && win.livePreview
                                hint: "Seconds between previews. Elapsed time rather than a frame count: 30 frames is one preview a second on a small clip and one every 51 seconds at 8K, which is backwards — the long render is the one worth watching."
                                Slider {
                                    Layout.fillWidth: true
                                    from: 1; to: 15; stepSize: 1
                                    value: win.livePreviewEvery
                                    onMoved: win.livePreviewEvery = value
                                }
                                Text {
                                    text: win.livePreviewEvery.toFixed(0) + "s"
                                    color: Theme.text
                                    font.pixelSize: Theme.fontM
                                    Layout.preferredWidth: 32
                                }
                            }

                            Row2 {
                                label: "Depth model"
                                hint: win.depthChoiceDetail(win.depthBackend,
                                                            win.depthModel)
                                ComboBox {
                                    id: depthBox
                                    Layout.fillWidth: true
                                    textRole: "text"
                                    Layout.minimumWidth: 260
                                    model: win.depthChoices
                                    currentIndex: win.depthChoiceIndex(
                                        win.depthBackend, win.depthModel)
                                    onActivated: {
                                        var c = win.depthChoices[currentIndex]
                                        if (!win.backendUsable(c.backend)) {
                                            // Put the reason in the log here,
                                            // rather than listing every
                                            // unavailable backend at startup:
                                            // the selection is about to snap
                                            // back, and that is what wants
                                            // explaining.
                                            app.explainBackend(c.backend)
                                            currentIndex = win.depthChoiceIndex(
                                                win.depthBackend, win.depthModel)
                                            return
                                        }
                                        win.depthBackend = c.backend
                                        // Variant left alone for the entries
                                        // that have none, so switching to
                                        // Depth Pro and back does not lose it.
                                        if (c.model !== "")
                                            win.depthModel = c.model
                                    }

                                    // Unavailable backends stay listed but
                                    // cannot be chosen. Hiding them would
                                    // leave no clue they exist; letting them
                                    // be chosen just moves the failure later.
                                    delegate: ItemDelegate {
                                        width: depthBox.width
                                        enabled: win.backendUsable(modelData.backend)
                                        highlighted:
                                            depthBox.highlightedIndex === index
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
                                                    modelData.backend)
                                                color: Theme.warn
                                                font.pixelSize: Theme.fontS
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                        }
                                    }

                                    // Same reason as the quality preset: the
                                    // control severs its own binding when
                                    // activated, so re-apply it explicitly.
                                    Connections {
                                        target: win
                                        function onDepthBackendChanged() {
                                            depthBox.currentIndex =
                                                win.depthChoiceIndex(
                                                    win.depthBackend,
                                                    win.depthModel)
                                        }
                                        function onDepthModelChanged() {
                                            depthBox.currentIndex =
                                                win.depthChoiceIndex(
                                                    win.depthBackend,
                                                    win.depthModel)
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
