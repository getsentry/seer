import unittest
from unittest.mock import MagicMock, patch
from src.seer.automation.autofix.event_manager import EventManager
from src.seer.automation.autofix.types import Step, AutofixStatus

class TestEventManager(unittest.TestCase):
    def setUp(self):
        self.event_manager = EventManager()

    @patch('src.seer.automation.autofix.event_manager.EventManager.state')
    def test_restart_step(self, mock_state):
        # Arrange
        mock_cur = MagicMock()
        mock_state.update.return_value.__enter__.return_value = mock_cur
        
        step = Step(id="test_step", name="Test Step")
        mock_cur.steps = [step]
        mock_cur.status = AutofixStatus.FAILED

        # Act
        self.event_manager.restart_step(step)

        # Assert
        self.assertEqual(mock_cur.steps[0].progress, [])
        self.assertIsNone(mock_cur.steps[0].completedMessage)
        self.assertEqual(mock_cur.status, AutofixStatus.PROCESSING)
        mock_cur.mark_triggered.assert_called_once()

if __name__ == '__main__':
    unittest.main()