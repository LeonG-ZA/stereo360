import QtQuick
import QtQuick.Layouts
import Stereo360

// A titled panel. `collapsible` keeps the rarely-touched options present but
// out of the way -- the settings surface is large enough that showing all of
// it at once is what makes a tool look intimidating.
Rectangle {
    id: root

    property string title: ""
    property string subtitle: ""
    property bool collapsible: false
    property bool expanded: true
    default property alias content: body.data

    Layout.fillWidth: true
    implicitHeight: column.implicitHeight + Theme.pad * 2
    color: Theme.surface
    radius: Theme.radius
    border.width: 1
    border.color: Theme.border

    ColumnLayout {
        id: column
        anchors.fill: parent
        anchors.margins: Theme.pad
        spacing: root.expanded ? Theme.gap : 0

        // A plain Item, not the RowLayout itself, so the click target can be
        // anchored over the whole header. A layout manages its children's
        // geometry, so anchoring one of them is undefined behaviour and Qt
        // warns about it at every polish.
        Item {
            Layout.fillWidth: true
            implicitHeight: header.implicitHeight
            visible: root.title !== ""

            RowLayout {
                id: header
                anchors.fill: parent
                spacing: 8

                Text {
                    text: root.title
                    color: Theme.text
                    font.pixelSize: Theme.fontM
                    font.weight: Font.DemiBold
                }
                Text {
                    text: root.subtitle
                    color: Theme.textFaint
                    font.pixelSize: Theme.fontS
                    Layout.fillWidth: true
                    elide: Text.ElideRight
                }
                Text {
                    visible: root.collapsible
                    text: root.expanded ? "–" : "+"
                    color: Theme.textDim
                    font.pixelSize: Theme.fontL
                }
            }

            MouseArea {
                anchors.fill: parent
                enabled: root.collapsible
                cursorShape: Qt.PointingHandCursor
                onClicked: root.expanded = !root.expanded
            }
        }

        ColumnLayout {
            id: body
            Layout.fillWidth: true
            spacing: Theme.gap
            visible: root.expanded
        }
    }

    Behavior on implicitHeight {
        NumberAnimation { duration: 120; easing.type: Easing.OutCubic }
    }
}
