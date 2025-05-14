import datetime
import unittest
from unittest.mock import patch

from seer.automation.codegen.bank_account import (
    AccountLocked,
    BankAccount,
    DailyLimitExceeded,
    InsufficientFunds,
)


class TestBankAccount(unittest.TestCase):
    def setUp(self):
        self.account = BankAccount(owner="Test Owner", balance=1000.0)
        self.second_account = BankAccount(owner="Second Owner", balance=500.0)

    def test_initialization(self):
        """Test that a bank account is initialized with correct values."""
        account = BankAccount(owner="John Doe", balance=100.0, interest_rate=0.05, daily_limit=500)
        self.assertEqual(account.owner, "John Doe")
        self.assertEqual(account.balance, 100.0)
        self.assertEqual(account.interest_rate, 0.05)
        self.assertEqual(account.daily_limit, 500)
        self.assertFalse(account.locked)
        self.assertEqual(account.transactions, [])
        self.assertEqual(account.withdrawals_today, 0)
        self.assertIsNone(account.last_withdrawal_date)

    def test_deposit_valid_amount(self):
        """Test depositing a valid amount."""
        initial_balance = self.account.balance
        amount = 500.0
        new_balance = self.account.deposit(amount)
        
        self.assertEqual(new_balance, initial_balance + amount)
        self.assertEqual(self.account.balance, initial_balance + amount)
        self.assertEqual(len(self.account.transactions), 1)
        self.assertEqual(self.account.transactions[0][0], "deposit")
        self.assertEqual(self.account.transactions[0][1], amount)

    def test_deposit_invalid_amount(self):
        """Test depositing an invalid amount raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.deposit(-100)

        with self.assertRaises(ValueError):
            self.account.deposit(0)

    def test_deposit_locked_account(self):
        """Test depositing to a locked account raises AccountLocked."""
        self.account.lock()
        with self.assertRaises(AccountLocked):
            self.account.deposit(100)

    def test_withdraw_valid_amount(self):
        """Test withdrawing a valid amount."""
        initial_balance = self.account.balance
        amount = 500.0
        new_balance = self.account.withdraw(amount)
        
        self.assertEqual(new_balance, initial_balance - amount)
        self.assertEqual(self.account.balance, initial_balance - amount)
        self.assertEqual(len(self.account.transactions), 1)
        self.assertEqual(self.account.transactions[0][0], "withdraw")
        self.assertEqual(self.account.transactions[0][1], amount)

    def test_withdraw_invalid_amount(self):
        """Test withdrawing an invalid amount raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.withdraw(-100)

        with self.assertRaises(ValueError):
            self.account.withdraw(0)

    def test_withdraw_insufficient_funds(self):
        """Test withdrawing more than balance raises InsufficientFunds."""
        with self.assertRaises(InsufficientFunds):
            self.account.withdraw(self.account.balance + 1)

    def test_withdraw_locked_account(self):
        """Test withdrawing from a locked account raises AccountLocked."""
        self.account.lock()
        with self.assertRaises(AccountLocked):
            self.account.withdraw(100)

    def test_withdraw_exceeds_daily_limit(self):
        """Test withdrawing more than daily limit raises DailyLimitExceeded."""
        self.account.withdraw(self.account.daily_limit / 2)
        # Try to withdraw more than remaining daily limit
        with self.assertRaises(DailyLimitExceeded):
            self.account.withdraw(self.account.daily_limit / 2 + 1)

    @patch('datetime.date')
    def test_reset_daily_limit(self, mock_date):
        """Test daily withdrawal limit resets on a new day."""
        today = datetime.date(2023, 1, 1)
        tomorrow = datetime.date(2023, 1, 2)
        
        # Set up "today"
        mock_date.today.return_value = today
        self.account.withdraw(500)
        self.assertEqual(self.account.withdrawals_today, 500)
        self.assertEqual(self.account.last_withdrawal_date, today)
        
        # Change to "tomorrow"
        mock_date.today.return_value = tomorrow
        self.account.withdraw(200)
        self.assertEqual(self.account.withdrawals_today, 200)
        self.assertEqual(self.account.last_withdrawal_date, tomorrow)

    def test_transfer_successful(self):
        """Test transferring money between accounts."""
        initial_balance = self.account.balance
        target_initial_balance = self.second_account.balance
        amount = 300.0
        
        new_balance = self.account.transfer(self.second_account, amount)
        
        self.assertEqual(new_balance, initial_balance - amount)
        self.assertEqual(self.account.balance, initial_balance - amount)
        self.assertEqual(self.second_account.balance, target_initial_balance + amount)
        self.assertEqual(len(self.account.transactions), 1)
        self.assertEqual(self.account.transactions[0][0], "transfer")
        self.assertEqual(self.account.transactions[0][1], amount)
        self.assertEqual(self.account.transactions[0][3], self.second_account.owner)

    def test_transfer_invalid_target(self):
        """Test transferring to a non-BankAccount object raises TypeError."""
        with self.assertRaises(TypeError):
            self.account.transfer("not an account", 100)

    def test_apply_interest(self):
        """Test applying interest to the account."""
        initial_balance = self.account.balance
        interest_rate = self.account.interest_rate
        expected_interest = initial_balance * interest_rate
        
        earned_interest = self.account.apply_interest()
        
        self.assertEqual(earned_interest, expected_interest)
        self.assertEqual(self.account.balance, initial_balance + expected_interest)
        self.assertEqual(len(self.account.transactions), 1)
        self.assertEqual(self.account.transactions[0][0], "interest")
        self.assertEqual(self.account.transactions[0][1], expected_interest)

    def test_get_transaction_history(self):
        """Test retrieving transaction history with limit."""
        # Add multiple transactions
        self.account.deposit(100)
        self.account.withdraw(50)
        self.account.deposit(200)
        self.account.withdraw(75)
        
        # Get limited history (default limit=10)
        history = self.account.get_transaction_history()
        self.assertEqual(len(history), 4)
        
        # Get limited history with custom limit
        history = self.account.get_transaction_history(limit=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0][0], "withdraw")
        self.assertEqual(history[0][1], 75)
        self.assertEqual(history[1][0], "deposit")
        self.assertEqual(history[1][1], 200)