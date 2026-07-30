import QtQuick
import QtQuick.Layouts
import Stereo360

// Label on the left, control on the right, with an optional hint underneath.
// A fixed label column is what stops a settings panel reading as a jumble.
ColumnLayout {
    id: root

    property string label: ""
    property string hint: ""
    property int labelWidth: 96
    default property alias control: holder.data

    Layout.fillWidth: true
    spacing: 4

    RowLayout {
        Layout.fillWidth: true
        spacing: 10

        Text {
            text: root.label
            color: Theme.textDim
            font.pixelSize: Theme.fontM
            Layout.preferredWidth: root.labelWidth
            Layout.alignment: Qt.AlignVCenter
            elide: Text.ElideRight
        }

        RowLayout {
            id: holder
            Layout.fillWidth: true
            spacing: 8
        }
    }

    Text {
        visible: root.hint !== ""
        text: root.hint
        color: Theme.textFaint
        font.pixelSize: Theme.fontS
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
        Layout.leftMargin: root.labelWidth + 10
    }
}
