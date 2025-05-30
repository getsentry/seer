import datetime


class InsufficientFunds(Exception):
    pass


class AccountLocked(Exception):
    pass


class DailyLimitExceeded(Exception):
    pass


class BankAccount:
    def __init__(self, owner, balance=0.0, interest_rate=0.01, daily_limit=1000):
        self.owner = owner
        self.balance = float(balance)
        self.interest_rate = interest_rate
        self.daily_limit = daily_limit
        self.locked = False
        self.transactions = []
        self.withdrawals_today = 0
        self.last_withdrawal_date = None

    def _check_locked(self):
        if self.locked:
            raise AccountLocked("Account is locked")

    def _reset_daily_limit_if_needed(self):
        today = datetime.date.today()
        if self.last_withdrawal_date != today:
            self.withdrawals_today = 0
            self.last_withdrawal_date = today

    def deposit(self, amount):
        self._check_locked()
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self.balance += amount
        self.transactions.append(("deposit", amount, datetime.datetime.now()))
        return self.balance

    def withdraw(self, amount):
        self._check_locked()
        self._reset_daily_limit_if_needed()
        if amount <= 0:
            raise ValueError("Withdrawal must be positive")
        if amount > self.balance:
            raise InsufficientFunds("Insufficient funds")
        if self.withdrawals_today + amount > self.daily_limit:
            raise DailyLimitExceeded("Daily withdrawal limit exceeded")
        self.balance -= amount
        self.withdrawals_today += amount
        self.transactions.append(("withdraw", amount, datetime.datetime.now()))
        return self.balance

    def transfer(self, target, amount):
        if not isinstance(target, BankAccount):
            raise TypeError("Target must be a BankAccount")
        self.withdraw(amount)
        target.deposit(amount)
        self.transactions.append(("transfer", amount, datetime.datetime.now(), target.owner))
        return self.balance

    def apply_interest(self):
        self._check_locked()
        interest = self.balance * self.interest_rate
        self.balance += interest
        self.transactions.append(("interest", interest, datetime.datetime.now()))
        return interest

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

    def get_transaction_history(self, limit=10):
        return self.transactions[-limit:]
