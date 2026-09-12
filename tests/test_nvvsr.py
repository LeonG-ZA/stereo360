"""Who is offered NVIDIA VSR, and what they agreed to.

Two rules here are easy to get subtly wrong and impossible to notice from a
working machine, because a developer with an RTX card sees the happy path
whichever way the logic falls.

The first is the tensor-core rule. The SDK needs them, they arrived with
Turing, and Turing is compute capability 7.5 -- so `>= 7.5` looks right and
is not: the GTX 1650 and 1660 are Turing parts *without* tensor cores and
report 7.5 anyway. A bare capability test offers those owners a 490 MB
download that cannot run.

The second is the Linux driver floor, which NVIDIA states as branch minimums
(570.190, 580.82, 590.44) rather than one number. Read as "newer than
570.190" it would wave through a 580.0 driver that NVIDIA does not cover.

The consent tests pin the property that makes the dialog worth building: it
records *which* agreement was accepted, so an updated one asks again.
"""

import platform

import pytest

from stereo360 import nvvsr


def _gpu(name="NVIDIA GeForce RTX 4070", cap=(8, 9), driver=(571, 96)):
    return nvvsr.Gpu(name=name, capability=cap, driver=driver)


class TestTensorCores:
    def test_an_rtx_20_series_card_qualifies(self):
        """Turing with tensor cores is the floor NVIDIA states."""
        assert _gpu("NVIDIA GeForce RTX 2060", (7, 5)).has_tensor_cores

    def test_anything_newer_qualifies(self):
        for name, cap in (("NVIDIA GeForce RTX 3080", (8, 6)),
                          ("NVIDIA GeForce RTX 4090", (8, 9)),
                          ("NVIDIA GeForce RTX 5070 Ti", (12, 0)),
                          ("NVIDIA H100 PCIe", (9, 0))):
            assert _gpu(name, cap).has_tensor_cores, name

    def test_a_gtx_16_series_does_not_despite_reporting_7_5(self):
        """The case a bare capability check gets wrong.

        These are Turing dies with the tensor cores fused off. They report
        7.5 like an RTX 2060 and cannot run the SDK.
        """
        for name in ("NVIDIA GeForce GTX 1660 SUPER",
                     "NVIDIA GeForce GTX 1650",
                     "NVIDIA GeForce GTX 1660 Ti"):
            g = _gpu(name, (7, 5))
            assert g.capability >= nvvsr.MIN_CAPABILITY, "still 7.5"
            assert not g.has_tensor_cores, name

    def test_a_gtx_10_series_does_not(self):
        assert not _gpu("NVIDIA GeForce GTX 1080 Ti", (6, 1)).has_tensor_cores


class TestDriverFloor:
    def test_windows_uses_one_number(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        assert _gpu(driver=(570, 65)).driver_ok
        assert _gpu(driver=(610, 74)).driver_ok
        assert not _gpu(driver=(570, 64)).driver_ok
        assert not _gpu(driver=(566, 36)).driver_ok

    def test_linux_is_checked_against_its_own_branch(self, monkeypatch):
        """580.0 is numerically past 570.190 and is still too old.

        NVIDIA lists 570.190+, 580.82+ and 590.44+, which are three floors
        rather than a range, so each branch is judged on its own.
        """
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert _gpu(driver=(570, 190)).driver_ok
        assert not _gpu(driver=(570, 189)).driver_ok
        assert _gpu(driver=(580, 82)).driver_ok
        assert not _gpu(driver=(580, 0)).driver_ok, "a newer branch, older driver"
        assert _gpu(driver=(590, 44)).driver_ok
        assert not _gpu(driver=(590, 43)).driver_ok

    def test_a_branch_newer_than_any_listed_is_allowed(self, monkeypatch):
        """The table cannot list drivers that do not exist yet."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert _gpu(driver=(600, 0)).driver_ok

    def test_a_branch_older_than_any_listed_is_not(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert not _gpu(driver=(560, 99)).driver_ok


class TestOffering:
    def test_no_nvidia_gpu_is_refused_with_a_reason(self, monkeypatch):
        monkeypatch.setattr(nvvsr, "_smi", lambda: None)
        assert not nvvsr.supported()
        assert "NVIDIA GPU" in nvvsr.why_not()

    def test_the_reason_names_the_card_it_found(self, monkeypatch):
        monkeypatch.setattr(
            nvvsr, "_smi",
            lambda: "NVIDIA GeForce GTX 1660 SUPER, 7.5, 571.96\n")
        assert not nvvsr.supported()
        assert "GTX 1660 SUPER" in nvvsr.why_not()
        assert "RTX 20 series" in nvvsr.why_not()

    def test_a_supported_card_has_nothing_to_explain(self, monkeypatch):
        monkeypatch.setattr(
            nvvsr, "_smi",
            lambda: "NVIDIA GeForce RTX 4070, 8.9, 571.96\n")
        assert nvvsr.supported()
        assert nvvsr.why_not() is None

    def test_smi_missing_is_not_an_error(self, monkeypatch):
        """A machine with no driver at all must answer, not raise."""
        def boom(*a, **k):
            raise FileNotFoundError("nvidia-smi")
        monkeypatch.setattr(nvvsr.subprocess, "run", boom)
        assert nvvsr.gpu() is None
        assert not nvvsr.supported()


class TestConsent:
    def test_nothing_is_accepted_to_begin_with(self, tmp_path):
        assert not nvvsr.accepted(where=str(tmp_path / "none"))

    def test_recording_then_asking_agrees(self, tmp_path):
        where = str(tmp_path / "models" / ".accepted")
        nvvsr.record("THE AGREEMENT", where=where)
        assert nvvsr.accepted("THE AGREEMENT", where=where)

    def test_a_changed_agreement_asks_again(self, tmp_path):
        """The reason consent is keyed by text rather than a flag.

        NVIDIA reserves the right to update the agreement; a tick from last
        year should not carry over to terms nobody has read.
        """
        where = str(tmp_path / "models" / ".accepted")
        nvvsr.record("VERSION ONE", where=where)
        assert nvvsr.accepted("VERSION ONE", where=where)
        assert not nvvsr.accepted("VERSION TWO", where=where)

    def test_both_are_remembered_once_both_are_accepted(self, tmp_path):
        where = str(tmp_path / "models" / ".accepted")
        nvvsr.record("VERSION ONE", where=where)
        nvvsr.record("VERSION TWO", where=where)
        assert nvvsr.accepted("VERSION ONE", where=where)
        assert nvvsr.accepted("VERSION TWO", where=where)

    def test_wrapping_does_not_count_as_a_change(self, tmp_path):
        """The text is hashed on its words, not its line breaks.

        A dialog that re-wraps to the window width would otherwise invalidate
        a consent that was given to exactly these terms.
        """
        where = str(tmp_path / "models" / ".accepted")
        nvvsr.record("the agreement\nwrapped one way", where=where)
        assert nvvsr.accepted("the   agreement wrapped\n\none way", where=where)

    def test_offline_honours_a_consent_already_held(self, tmp_path):
        """Asked with no text -- which is what a machine that cannot reach
        NVIDIA can ask -- any recorded acceptance counts."""
        where = str(tmp_path / "models" / ".accepted")
        assert not nvvsr.accepted(where=where)
        nvvsr.record("ANYTHING", where=where)
        assert nvvsr.accepted(where=where)


class TestConsentIsToTheRealAgreement:
    """`accepted()` with no text answers a much weaker question than it
    looks, and something has to make sure nothing important asks it.

    It means "has anything ever been accepted". A marker left by anything at
    all then unlocks the model -- and that happened: a test wrote consent to
    the string "THE TERMS" into the real models/ directory, `present()` asked
    the textless form, and NVIDIA VSR was offered as though its licence had
    been read. `consented()` is the question that had to be asked instead.
    """

    def test_a_foreign_marker_does_not_unlock_it(self, tmp_path, monkeypatch):
        from stereo360 import nvvsr

        marker = str(tmp_path / "models" / ".accepted")
        monkeypatch.setattr(nvvsr, "CONSENT", marker)
        monkeypatch.setattr(nvvsr, "agreement", lambda: "THE REAL AGREEMENT")
        nvvsr.record("SOMETHING ELSE ENTIRELY", where=marker)

        assert nvvsr.accepted(where=marker), "the weak question says yes"
        assert not nvvsr.consented(), "and the one that matters says no"

    def test_the_matching_agreement_does(self, tmp_path, monkeypatch):
        from stereo360 import nvvsr

        marker = str(tmp_path / "models" / ".accepted")
        monkeypatch.setattr(nvvsr, "CONSENT", marker)
        monkeypatch.setattr(nvvsr, "agreement", lambda: "THE REAL AGREEMENT")
        nvvsr.record("THE REAL AGREEMENT", where=marker)
        assert nvvsr.consented()

    def test_an_unreadable_agreement_is_a_refusal(self, tmp_path, monkeypatch):
        """Terms that cannot be shown cannot have been agreed to, which is
        what the dialog says too by leaving Accept disabled."""
        from stereo360 import nvvsr

        marker = str(tmp_path / "models" / ".accepted")
        monkeypatch.setattr(nvvsr, "CONSENT", marker)
        monkeypatch.setattr(nvvsr, "agreement", lambda: None)
        nvvsr.record("ANYTHING", where=marker)
        assert not nvvsr.consented()

    def test_the_model_is_not_offered_without_it(self, monkeypatch):
        """`present()` is what the probe filters on, so this is the gate."""
        from stereo360 import nvvsr, upscalers

        monkeypatch.setattr(nvvsr, "installed", lambda: True)
        monkeypatch.setattr(nvvsr, "consented", lambda: False)
        assert not upscalers.present(upscalers.get("nvvsr_ultra"))
        monkeypatch.setattr(nvvsr, "consented", lambda: True)
        assert upscalers.present(upscalers.get("nvvsr_ultra"))


class TestControllerSlots:
    """The three slots QML calls, exercised through a real Controller.

    Parsing is not evidence: a @Slot with the wrong result type compiles and
    then hands QML something it cannot read.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def ctrl(cls):
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.instance() or QCoreApplication([])
        from stereo360_ui.controller import Controller

        return Controller()

    def test_status_is_a_map_qml_can_read(self, ctrl):
        d = ctrl.nvvsrStatus()
        assert isinstance(d, dict)
        assert d["name"] == "NVIDIA VSR"
        assert isinstance(d["supported"], bool)

    def test_an_unreadable_agreement_is_an_empty_string_not_none(self, ctrl,
                                                                monkeypatch):
        """QML gets "" and disables Accept; None would arrive as undefined."""
        from stereo360 import nvvsr

        monkeypatch.setattr(nvvsr, "agreement", lambda: None)
        assert ctrl.nvvsrAgreement() == ""

    def test_accepting_nothing_is_refused(self, ctrl, monkeypatch, tmp_path):
        """A dialog that could not show the terms must not record consent."""
        from stereo360 import nvvsr

        marker = str(tmp_path / "models" / ".accepted")
        monkeypatch.setattr(nvvsr, "CONSENT", marker)
        assert ctrl.nvvsrAccept("") is False
        assert ctrl.nvvsrAccept("   \n ") is False
        assert not nvvsr.accepted(where=marker)

    def test_accepting_real_text_records_that_text(self, ctrl, monkeypatch,
                                                   tmp_path):
        from stereo360 import nvvsr

        marker = str(tmp_path / "models" / ".accepted")
        monkeypatch.setattr(nvvsr, "CONSENT", marker)
        assert ctrl.nvvsrAccept("THE TERMS") is True
        assert nvvsr.accepted("THE TERMS", where=marker)
        assert not nvvsr.accepted("OTHER TERMS", where=marker)


class TestTheInstallSlot:
    """The download itself, which must not start where it cannot finish."""

    @pytest.fixture(scope="class")
    @classmethod
    def ctrl(cls):
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.instance() or QCoreApplication([])
        from stereo360_ui.controller import Controller

        return Controller()

    def test_it_refuses_on_a_machine_that_cannot_run_it(self, ctrl,
                                                        monkeypatch):
        """A 490 MB download for a card without tensor cores is the one
        outcome the detection exists to prevent."""
        from stereo360 import nvvsr

        monkeypatch.setattr(nvvsr, "_smi", lambda: None)
        said = []
        ctrl.logged.connect(lambda level, text: said.append((level, text)))
        ctrl.nvvsrInstall()
        assert not ctrl.nvvsrFetching
        assert said and said[-1][0] == "warn"
        assert "NVIDIA GPU" in said[-1][1]

    def test_it_names_nvidias_index_and_not_pypi(self, ctrl, monkeypatch):
        """PyPI carries a 2.7 KB stub; the wheels are on NVIDIA's own index.
        Installing from the default index would appear to succeed and leave
        nothing importable."""
        from stereo360 import nvvsr

        monkeypatch.setattr(
            nvvsr, "_smi",
            lambda: "NVIDIA GeForce RTX 4070, 8.9, 571.96")
        started = {}
        monkeypatch.setattr(ctrl._nvvsr_proc, "start",
                            lambda: started.setdefault("yes", True))
        ctrl.nvvsrInstall()
        argv = ctrl._nvvsr_proc.arguments()
        assert started, "it did not start"
        assert "--extra-index-url" in argv
        assert nvvsr.INDEX in argv
        assert nvvsr.PACKAGE in argv
        assert "--no-input" in argv, "nothing here can answer a pip prompt"
        ctrl._nvvsr_fetching = False


def test_the_label_is_the_one_asked_for():
    """Named for the product, not the SDK it arrives in."""
    assert nvvsr.NAME == "NVIDIA VSR"


def test_describe_answers_every_question_the_ui_asks(monkeypatch):
    monkeypatch.setattr(nvvsr, "_smi",
                        lambda: "NVIDIA GeForce RTX 4070, 8.9, 571.96\n")
    d = nvvsr.describe()
    for key in ("code", "name", "desc", "supported", "installed", "accepted",
                "why_not", "gpu", "package", "index", "agreement_url"):
        assert key in d, key
    assert d["index"] == "https://pypi.nvidia.com", "PyPI carries only a stub"
