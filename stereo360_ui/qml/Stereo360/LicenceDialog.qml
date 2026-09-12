import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Stereo360

// Someone else's licence, presented before their software is used.
//
// Accept stays disabled until the text has actually been scrolled to the end.
// That is the point of the dialog rather than decoration on it: a tick box
// beside an unread wall of text records a click, and this records that the
// terms were at least put in front of someone.
//
// `text` is the agreement read out of the installed wheel. It is never
// synthesised or summarised here -- if the caller could not read the real
// thing it passes an empty string, and the dialog says so and refuses to
// offer Accept at all.
Dialog {
    id: root

    property string vendor: "NVIDIA"
    property string product: "NVIDIA VSR"
    property string text: ""
    property url moreInfo: ""

    // Emitted with the exact text that was on screen, so the caller records
    // consent against *this* agreement rather than a bare flag. An updated
    // one then asks again instead of passing under last year's tick.
    //
    // Not called `accepted`: Dialog already has a signal by that
    // name, so overriding it both warns and fires twice, since
    // `accept()` below raises the original.
    signal agreed(string agreementText)

    readonly property bool readable: text.trim().length > 0
    readonly property bool atEnd:
        !readable ? false
                  : body.contentHeight <= body.height + 2
                    || body.contentY >= body.contentHeight - body.height - 2

    title: qsTr("%1 licence agreement").arg(vendor)
    modal: true
    closePolicy: Popup.NoAutoClose
    standardButtons: Dialog.NoButton
    anchors.centerIn: Overlay.overlay
    width: Math.min(parent ? parent.width - Theme.gap * 4 : 820, 900)
    height: Math.min(parent ? parent.height - Theme.gap * 4 : 620, 720)

    background: Rectangle {
        color: Theme.surface
        border.color: Theme.border
        radius: Theme.radius
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: Theme.gap

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            color: Theme.textDim
            font.pixelSize: Theme.fontS
            text: root.readable
                  ? qsTr("%1 is %2's software, not part of this application. "
                         + "Read the agreement below and accept it to use %1. "
                         + "You are only asked once.")
                    .arg(root.product).arg(root.vendor)
                  : qsTr("The agreement could not be read, so it cannot be "
                         + "shown. %1 will stay unavailable.").arg(root.product)
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: Theme.bg
            border.color: Theme.border
            radius: Theme.radius / 2
            visible: root.readable

            Flickable {
                id: body
                anchors.fill: parent
                anchors.margins: Theme.pad
                clip: true
                contentWidth: width
                contentHeight: agreement.implicitHeight
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOn }

                TextEdit {
                    id: agreement
                    width: body.width - Theme.pad
                    text: root.text
                    // Selectable so the terms can be copied out, read-only so
                    // the thing being accepted cannot be edited first.
                    readOnly: true
                    selectByMouse: true
                    wrapMode: TextEdit.Wrap
                    color: Theme.text
                    font.pixelSize: Theme.fontS
                    font.family: "monospace"
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.gap

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                color: root.atEnd ? Theme.textFaint : Theme.warn
                font.pixelSize: Theme.fontS
                visible: root.readable
                text: root.atEnd ? qsTr("You have reached the end.")
                                 : qsTr("Scroll to the end to continue.")
            }

            Button {
                text: qsTr("Open in browser")
                visible: root.moreInfo != ""
                onClicked: Qt.openUrlExternally(root.moreInfo)
            }

            Button {
                text: qsTr("Decline")
                onClicked: root.reject()
            }

            Button {
                text: qsTr("Accept")
                enabled: root.readable && root.atEnd
                highlighted: enabled
                onClicked: {
                    root.agreed(root.text)
                    root.accept()
                }
            }
        }
    }
}
