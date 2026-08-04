// Lets the Repeater delegates below reach `root` and `strip` by id, which is
// otherwise undefined behaviour inside a component -- and makes `index` and
// `modelData` explicit rather than injected.
pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Stereo360

// Which way the VR180 field points, dragged on a picture of the source.
//
// A VR180 file keeps half the sphere and throws the rest away, so the one
// question it asks that a 360 render never does is *which* half. Answering it
// with a number in degrees means guessing what is at 40 degrees; answering it
// by dragging a band across the frame does not.
//
// The whole control is a column range with wraparound -- yaw resamples
// nothing -- which is why the result panel can update live off a single
// thumbnail with no rendering behind it.
ColumnLayout {
    id: root

    property string source: ""
    property real yaw: 0
    property bool loading: false

    //: Emitted with the new yaw in degrees, normalised to (-180, 180].
    signal yawMoved(real degrees)

    spacing: 8

    // Fraction across the image, 0 at the left edge, of the field's centre.
    // Yaw is measured from the middle of the equirect, which is where the
    // camera was pointing when it stitched.
    readonly property real centreFrac:
        ((((yaw + 180) % 360) + 360) % 360) / 360
    //: Half the field, as a fraction of the sphere: 90 of 360 degrees.
    readonly property real halfFrac: 0.25
    //: Left edge of the crop, wrapped into the image. Must agree with
    //: `pipeline.vr180_crop`, which is what the render actually does -- a
    //: picker that points somewhere other than the crop is a bug you would
    //: otherwise only find in a headset. Asserted against it in test_ui.py,
    //: which is why the selftest dumps this.
    readonly property real startFrac: {
        var f = (centreFrac - halfFrac) % 1
        return f < 0 ? f + 1 : f
    }

    function normalise(deg) {
        var d = ((deg + 180) % 360 + 360) % 360 - 180
        return d === -180 ? 180 : d      // read as "behind you", not "-behind"
    }

    // The band as one or two rectangles in fractions of the width. Two when
    // the field straddles the seam at the back of the sphere, which is not an
    // edge case to avoid -- it is half of all the directions there are.
    readonly property var bandSegments: {
        var a = centreFrac - halfFrac
        var b = centreFrac + halfFrac
        if (a < 0)  return [{x: 0, w: b}, {x: a + 1, w: -a}]
        if (b > 1)  return [{x: a, w: 1 - a}, {x: 0, w: b - 1}]
        return [{x: a, w: 2 * halfFrac}]
    }

    // ---- the frame, with the band dragged across it ---------------------
    Item {
        id: strip
        Layout.fillWidth: true
        Layout.preferredHeight: width / 2      // an equirect is always 2:1

        Rectangle {
            anchors.fill: parent
            color: Theme.bg
            radius: 4
            border.width: 1
            border.color: Theme.border
            clip: true

            Image {
                id: frame
                anchors.fill: parent
                anchors.margins: 1
                source: root.source
                fillMode: Image.Stretch      // it is 2:1 and so is the box
                asynchronous: true
                cache: false
                visible: root.source !== ""
            }

            Text {
                anchors.centerIn: parent
                visible: root.source === ""
                text: root.loading ? "Reading the frame…"
                                   : "Choose an input video"
                color: Theme.textFaint
                font.pixelSize: Theme.fontS
            }

            // Everything outside the field, dimmed. Drawn as a full-width
            // veil with the band punched out of it rather than as two side
            // panels, so the wrapped case needs no special handling.
            Rectangle {
                anchors.fill: parent
                color: "#99000000"
            }

            Repeater {
                model: root.bandSegments
                Item {
                    required property var modelData
                    x: modelData.x * strip.width
                    width: modelData.w * strip.width
                    height: strip.height
                    clip: true
                    Image {
                        x: -parent.x
                        width: strip.width
                        height: strip.height
                        source: root.source
                        fillMode: Image.Stretch
                        asynchronous: true
                        cache: false
                        visible: root.source !== ""
                    }
                }
            }

            // The band's own outline, on top of the undimmed picture.
            Repeater {
                model: root.bandSegments
                Rectangle {
                    required property var modelData
                    x: modelData.x * strip.width
                    width: modelData.w * strip.width
                    height: strip.height
                    color: "transparent"
                    border.width: 2
                    border.color: Theme.accent
                }
            }

            // Where yaw is measured from: the middle of the equirect.
            Rectangle {
                x: strip.width / 2
                width: 1
                height: strip.height
                color: "#66ffffff"
            }
        }

        MouseArea {
            id: drag
            anchors.fill: parent
            cursorShape: Qt.SizeHorCursor
            hoverEnabled: false

            property real pressedX: 0
            property real pressedYaw: 0

            function yawAt(px) {
                return root.normalise(px / width * 360 - 180)
            }
            function inside(px) {
                var d = Math.abs(((px / width - root.centreFrac + 0.5) % 1
                                  + 1) % 1 - 0.5)
                return d <= root.halfFrac
            }

            // Press inside the band to nudge it, press outside to send it
            // there. Both then drag, so neither gesture has to be learned
            // before the other one works.
            onPressed: (mouse) => {
                pressedYaw = inside(mouse.x) ? root.yaw : yawAt(mouse.x)
                pressedX = mouse.x
                if (pressedYaw !== root.yaw)
                    root.yawMoved(pressedYaw)
            }
            onPositionChanged: (mouse) => {
                if (!pressed)
                    return
                var moved = (mouse.x - pressedX) / width * 360
                // Whole degrees: a 360 source has about 21 columns to the
                // degree, and nobody is choosing a direction to finer than
                // that. Keeps the readout from flickering while dragging.
                root.yawMoved(root.normalise(Math.round(pressedYaw + moved)))
            }
        }
    }

    // ---- what that leaves, and the readout -------------------------------
    RowLayout {
        Layout.fillWidth: true
        spacing: 10

        // The kept half as one picture. Earns its place precisely when the
        // band wraps: two bright stripes at opposite edges of the strip above
        // are the same view, and this is the only thing that shows it.
        Rectangle {
            Layout.preferredWidth: Theme.compact ? 92 : 108
            // Not `width`, which the layout derives from the line above --
            // that is a binding loop with a plausible-looking answer.
            Layout.preferredHeight: Layout.preferredWidth   // 180 of 360
            color: Theme.bg
            radius: 4
            border.width: 1
            border.color: Theme.border
            clip: true

            Repeater {
                // Two copies, offset by a full turn. The crop starts anywhere
                // in [0, 1), so at most one seam can fall inside the box and
                // the second copy is always the piece on the far side of it.
                model: root.source !== "" ? 2 : 0
                Image {
                    required property int index
                    width: parent.width * 2                // the whole sphere
                    height: parent.height
                    x: -root.startFrac * width + index * width
                    source: root.source
                    fillMode: Image.Stretch
                    asynchronous: true
                    cache: false
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4

            RowLayout {
                spacing: 8
                Text {
                    text: root.yaw === 0 ? "Straight ahead"
                        : root.yaw === 180 ? "Behind the camera"
                        : (root.yaw > 0 ? "+" : "−")
                          + Math.abs(root.yaw).toFixed(0) + "°"
                          + (root.yaw > 0 ? " right" : " left")
                    color: Theme.text
                    font.pixelSize: Theme.fontM
                    font.weight: Font.DemiBold
                }
                Button {
                    visible: root.yaw !== 0
                    flat: true
                    text: "Reset"
                    padding: 4
                    onClicked: root.yawMoved(0)
                }
            }

            Text {
                text: "Drag the band to choose which half of the sphere the "
                    + "file keeps. It costs nothing to move: the crop picks "
                    + "columns rather than rotating the picture."
                color: Theme.textFaint
                font.pixelSize: Theme.fontS
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }
}
