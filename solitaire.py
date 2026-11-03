"""
Klondike Solitaire - CSC200 Data Structures & Algorithms mid-term project.

Where each required data structure is used (and why):

  Array/List  -> Deck.cards          : fixed 52 cards, fast random access for shuffling
  Stack (LIFO)-> Stack               : foundation piles, waste pile, undo/redo history
  Queue (FIFO)-> Queue               : the stock (draw pile): first card in is first drawn
  Linked list -> TableauPile         : each tableau column; head of list = top card
  Dictionary  -> Game.piles          : pile name ("T3", "F1", "W") -> pile object
                 Game.location       : card key (suit, rank) -> pile name

Phases:  1 Card + Deck | 2 Tableau + Foundation | 3 Game logic | 4 Text UI
"""
from __future__ import annotations

import random
import sys
import time
from collections import deque
from dataclasses import dataclass

SUITS = ("♠", "♥", "♦", "♣")          # F1..F4 are tied to these, in this order
RED_SUITS = ("♥", "♦")
RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}
TABLEAU_NAMES = tuple(f"T{i}" for i in range(1, 8))
FOUNDATION_NAMES = tuple(f"F{i}" for i in range(1, 5))


# ======================================================================
# PHASE 1 - Card and Deck
# ======================================================================
class Card:
    """A playing card: suit, rank (1=Ace .. 13=King) and face-up/down state."""

    __slots__ = ("suit", "rank", "face_up")

    def __init__(self, suit: str, rank: int, face_up: bool = False):
        self.suit = suit
        self.rank = rank
        self.face_up = face_up

    @property
    def color(self) -> str:
        return "red" if self.suit in RED_SUITS else "black"

    @property
    def key(self) -> tuple[str, int]:
        """Hashable identity, used as the dictionary key for card locations."""
        return (self.suit, self.rank)

    @property
    def label(self) -> str:
        return f"{RANK_NAMES.get(self.rank, self.rank)}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.label}, {'up' if self.face_up else 'down'})"


class Deck:
    """52 cards held in a plain list (array)."""

    def __init__(self):
        self.cards: list[Card] = [Card(s, r) for s in SUITS for r in range(1, 14)]

    def shuffle(self, rng: random.Random | None = None) -> None:
        """Fisher-Yates shuffle: O(n), and every permutation is equally likely.

        Walk from the last position down; swap each card with a random card at
        or before it. Taking j from the *remaining* range (0..i) is what makes
        it unbiased - picking j from the whole deck every time would not be.
        """
        rng = rng or random
        for i in range(len(self.cards) - 1, 0, -1):
            j = rng.randint(0, i)
            self.cards[i], self.cards[j] = self.cards[j], self.cards[i]

    def deal_one(self) -> Card:
        return self.cards.pop()

    def __len__(self) -> int:
        return len(self.cards)


# ======================================================================
# PHASE 2 - Data structures: Stack, Queue, linked-list tableau pile
# ======================================================================
class Stack:
    """LIFO stack (backed by a list; push/pop at the end are O(1))."""

    def __init__(self):
        self._items: list = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        return self._items.pop()

    def peek(self):
        return self._items[-1] if self._items else None

    def is_empty(self) -> bool:
        return not self._items

    def clear(self) -> None:
        self._items.clear()

    def bottom_to_top(self) -> list:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class Queue:
    """FIFO queue (deque gives O(1) at both ends).

    enqueue/dequeue are the normal FIFO operations. push_front exists only so
    that UNDO of a draw can put a card back at the front of the line.
    """

    def __init__(self):
        self._items: deque = deque()

    def enqueue(self, item) -> None:
        self._items.append(item)

    def dequeue(self):
        return self._items.popleft()

    def push_front(self, item) -> None:
        self._items.appendleft(item)

    def is_empty(self) -> bool:
        return not self._items

    def to_list(self) -> list:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class _Node:
    __slots__ = ("card", "next")

    def __init__(self, card: Card, nxt: "_Node | None" = None):
        self.card = card
        self.next = nxt


class TableauPile:
    """A tableau column as a singly linked list.

    The HEAD is the TOP card (the one the player can touch). That choice makes
    the common operations cheap: pushing/popping the top is O(1), and moving a
    run of k cards only walks k nodes from the head instead of the whole pile.
    """

    def __init__(self):
        self.head: _Node | None = None
        self.size = 0

    def push(self, card: Card) -> None:
        self.head = _Node(card, self.head)
        self.size += 1

    def pop(self) -> Card:
        node = self.head
        self.head = node.next
        self.size -= 1
        return node.card

    def peek(self) -> Card | None:
        return self.head.card if self.head else None

    def top_cards(self, n: int) -> list[Card]:
        """The top n cards (top first) WITHOUT removing them."""
        out, node = [], self.head
        while node and len(out) < n:
            out.append(node.card)
            node = node.next
        return out

    def take(self, n: int) -> list[Card]:
        """Detach the top n cards (top first)."""
        return [self.pop() for _ in range(n)]

    def put(self, cards_top_first: list[Card]) -> None:
        """Attach a run of cards (given top first) on top of this pile."""
        for card in reversed(cards_top_first):   # bottom of the run goes on first
            self.push(card)

    def face_up_count(self) -> int:
        count, node = 0, self.head
        while node and node.card.face_up:
            count += 1
            node = node.next
        return count

    def bottom_to_top(self) -> list[Card]:
        return list(reversed(self.top_cards(self.size)))

    def __len__(self) -> int:
        return self.size


# ======================================================================
# PHASE 3 - Game logic
# ======================================================================
@dataclass
class Move:
    """One reversible action. The undo/redo stacks hold these."""

    kind: str                # "move" | "draw" | "recycle"
    src: str = ""
    dst: str = ""
    count: int = 0           # cards moved (or drawn)
    flipped: bool = False    # did this move turn a face-down card up?
    points: int = 0          # score change, so undo can subtract exactly this


class Game:
    def __init__(self, draw_count: int = 1, seed: int | None = None):
        self.draw_count = draw_count
        self.rng = random.Random(seed)
        self.new_game()

    # ---- setup ------------------------------------------------------
    def new_game(self) -> None:
        deck = Deck()
        deck.shuffle(self.rng)

        # Dictionary: pile name -> pile object (quick access by name).
        self.piles: dict[str, object] = {"W": Stack()}
        for name in TABLEAU_NAMES:
            self.piles[name] = TableauPile()
        for name in FOUNDATION_NAMES:
            self.piles[name] = Stack()
        self.stock = Queue()
        # Dictionary: card key -> pile name where the card currently is.
        self.location: dict[tuple[str, int], str] = {}

        # Deal: pile i gets i+1 cards, only the last one face up.
        for i, name in enumerate(TABLEAU_NAMES):
            for j in range(i + 1):
                card = deck.deal_one()
                card.face_up = (j == i)
                self.piles[name].push(card)
                self.location[card.key] = name
        # The remaining 24 cards form the stock, face down.
        while len(deck):
            card = deck.deal_one()
            self.stock.enqueue(card)
            self.location[card.key] = "S"

        self.undo_stack = Stack()
        self.redo_stack = Stack()
        self.score = 0
        self.started_at = time.time()
        self.finished_at: float | None = None

    # ---- read-only helpers -----------------------------------------
    @property
    def waste(self) -> Stack:
        return self.piles["W"]

    @property
    def moves(self) -> int:
        return len(self.undo_stack)

    def where(self, card: Card) -> str:
        return self.location[card.key]

    def is_won(self) -> bool:
        return sum(len(self.piles[f]) for f in FOUNDATION_NAMES) == 52

    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    @staticmethod
    def foundation_for(card: Card) -> str:
        return FOUNDATION_NAMES[SUITS.index(card.suit)]

    def snapshot(self) -> tuple:
        """Full comparable state (used by the tests to prove undo is exact)."""
        def row(cards):
            return tuple((c.key, c.face_up) for c in cards)
        return (
            tuple(row(self.piles[t].bottom_to_top()) for t in TABLEAU_NAMES),
            tuple(row(self.piles[f].bottom_to_top()) for f in FOUNDATION_NAMES),
            row(self.stock.to_list()),
            row(self.waste.bottom_to_top()),
            self.score,
        )

    # ---- rule checks -----------------------------------------------
    @staticmethod
    def is_valid_run(cards_top_first: list[Card]) -> bool:
        """Descending ranks, alternating colors (top card is the lowest)."""
        for upper, lower in zip(cards_top_first, cards_top_first[1:]):
            if lower.rank != upper.rank + 1 or lower.color == upper.color:
                return False
        return True

    def can_place_on_tableau(self, card: Card, name: str) -> bool:
        top = self.piles[name].peek()
        if top is None:
            return card.rank == 13                      # only a King on an empty column
        return top.face_up and top.color != card.color and top.rank == card.rank + 1

    def can_place_on_foundation(self, card: Card, name: str) -> bool:
        if card.suit != SUITS[int(name[1]) - 1]:        # F1=♠ F2=♥ F3=♦ F4=♣
            return False
        top = self.piles[name].peek()
        return card.rank == 1 if top is None else card.rank == top.rank + 1

    def _check_move(self, src: str, dst: str, n: int):
        """Validate a move. Returns (cards, resolved_dst, error_or_None)."""
        if src not in self.piles:
            return None, dst, f"Unknown source pile '{src}'."
        if dst != "F" and dst not in self.piles:
            return None, dst, f"Unknown destination pile '{dst}'."
        if dst == "W":
            return None, dst, "You can't move cards onto the waste pile."
        if n < 1:
            return None, dst, "Number of cards must be at least 1."

        src_pile = self.piles[src]
        if len(src_pile) == 0:
            return None, dst, f"{src} is empty."

        if src[0] == "T":
            if n > src_pile.face_up_count():
                return None, dst, f"Only {src_pile.face_up_count()} face-up card(s) in {src}."
            cards = src_pile.top_cards(n)
            if not self.is_valid_run(cards):
                return None, dst, "Those cards aren't a valid descending, alternating-color run."
        else:
            if n != 1:
                return None, dst, "Only one card at a time can leave the waste or a foundation."
            cards = [src_pile.peek()]

        lead = cards[-1]        # the card that will end up directly on the destination
        if dst == "F":
            dst = self.foundation_for(lead)
        if dst == src:
            return None, dst, "Source and destination are the same."

        if dst[0] == "F":
            if n != 1:
                return None, dst, "Only one card at a time can go to a foundation."
            if not self.can_place_on_foundation(lead, dst):
                return None, dst, f"{lead.label} can't go on {dst} right now."
        elif not self.can_place_on_tableau(lead, dst):
            return None, dst, f"{lead.label} can't go on {dst} (alternate colors, one rank lower; Kings on empty columns)."
        return cards, dst, None

    # ---- low-level mutations (no validation; shared by do/undo/redo) ---
    def _take(self, name: str, n: int) -> list[Card]:
        pile = self.piles[name]
        if isinstance(pile, TableauPile):
            return pile.take(n)
        return [pile.pop()]

    def _put(self, name: str, cards: list[Card]) -> None:
        pile = self.piles[name]
        if isinstance(pile, TableauPile):
            pile.put(cards)
        else:
            for card in reversed(cards):
                pile.push(card)
        for card in cards:
            self.location[card.key] = name

    def _execute(self, src: str, dst: str, n: int) -> bool:
        cards = self._take(src, n)
        flipped = False
        if src[0] == "T":                                # card flipping
            top = self.piles[src].peek()
            if top is not None and not top.face_up:
                top.face_up = True
                flipped = True
        self._put(dst, cards)
        return flipped

    def _reverse(self, m: Move) -> None:
        cards = self._take(m.dst, m.count)
        if m.flipped:
            self.piles[m.src].peek().face_up = False     # turn it back over first
        self._put(m.src, cards)

    def _do_draw(self, n: int) -> None:
        for _ in range(n):
            card = self.stock.dequeue()                  # FIFO: oldest card first
            card.face_up = True
            self.waste.push(card)
            self.location[card.key] = "W"

    def _undo_draw(self, n: int) -> None:
        for _ in range(n):
            card = self.waste.pop()
            card.face_up = False
            self.stock.push_front(card)                  # back to the front of the line
            self.location[card.key] = "S"

    def _do_recycle(self) -> None:
        popped = []
        while not self.waste.is_empty():
            popped.append(self.waste.pop())
        for card in reversed(popped):                    # waste bottom becomes next to draw
            card.face_up = False
            self.stock.enqueue(card)
            self.location[card.key] = "S"

    def _undo_recycle(self) -> None:
        while not self.stock.is_empty():
            card = self.stock.dequeue()
            card.face_up = True
            self.waste.push(card)
            self.location[card.key] = "W"

    def _record(self, m: Move) -> None:
        self.undo_stack.push(m)
        self.redo_stack.clear()                          # a new action ends the redo chain
        if self.is_won() and self.finished_at is None:
            self.finished_at = time.time()

    # ---- public actions ---------------------------------------------
    def move(self, src: str, dst: str, n: int = 1) -> tuple[bool, str]:
        src, dst = src.upper(), dst.upper()
        cards, dst, err = self._check_move(src, dst, n)
        if err:
            return False, err
        flipped = self._execute(src, dst, n)

        points = 0
        if flipped:
            points += 5
        if dst[0] == "F" and src[0] != "F":
            points += 10
        elif src == "W" and dst[0] == "T":
            points += 5
        elif src[0] == "F" and dst[0] == "T":
            points -= 15
        self.score += points

        self._record(Move("move", src, dst, n, flipped, points))
        names = " ".join(c.label for c in reversed(cards))
        return True, f"Moved {names} from {src} to {dst}."

    def draw(self) -> tuple[bool, str]:
        if self.stock.is_empty():
            if self.waste.is_empty():
                return False, "No cards left to draw."
            self._do_recycle()
            self._record(Move("recycle"))
            return True, "Stock was empty - waste turned over into a new stock."
        n = min(self.draw_count, len(self.stock))
        self._do_draw(n)
        self._record(Move("draw", count=n))
        return True, f"Drew {n} card(s)."

    def undo(self) -> tuple[bool, str]:
        if self.undo_stack.is_empty():
            return False, "Nothing to undo."
        m = self.undo_stack.pop()
        if m.kind == "move":
            self._reverse(m)
        elif m.kind == "draw":
            self._undo_draw(m.count)
        else:
            self._undo_recycle()
        self.score -= m.points
        self.redo_stack.push(m)
        self.finished_at = None
        return True, "Undone."

    def redo(self) -> tuple[bool, str]:
        if self.redo_stack.is_empty():
            return False, "Nothing to redo."
        m = self.redo_stack.pop()
        if m.kind == "move":
            self._execute(m.src, m.dst, m.count)
        elif m.kind == "draw":
            self._do_draw(m.count)
        else:
            self._do_recycle()
        self.score += m.points
        self.undo_stack.push(m)
        if self.is_won() and self.finished_at is None:
            self.finished_at = time.time()
        return True, "Redone."

    def hints(self) -> list[tuple[str, str, int]]:
        """All sensible legal moves as (src, dst, n); foundation moves first."""
        found: list[tuple[str, str, int]] = []
        for src in ("W",) + TABLEAU_NAMES:               # (skip foundation sources: rarely useful)
            pile = self.piles[src]
            if len(pile) == 0:
                continue
            max_n = pile.face_up_count() if src[0] == "T" else 1
            for n in range(1, max_n + 1):
                for dst in ("F",) + TABLEAU_NAMES:
                    cards, real_dst, err = self._check_move(src, dst, n)
                    if err:
                        continue
                    # Moving a King that already sits at the bottom of a column
                    # onto an empty column achieves nothing.
                    if (src[0] == "T" and real_dst[0] == "T" and n == len(pile)
                            and cards[-1].rank == 13):
                        continue
                    found.append((src, real_dst, n))
        found.sort(key=lambda h: h[1][0] != "F")
        return found

    # ---- text rendering (Phase 4) ----------------------------------
    def render(self, color: bool = True) -> str:
        def paint(text: str, card: Card | None) -> str:
            if color and card is not None and card.color == "red":
                return f"\033[31m{text}\033[0m"
            return text

        def cell(card: Card | None) -> str:
            if card is None:
                return "[   ]"
            if not card.face_up:
                return "[###]"
            return paint(f"[{card.label:<3}]", card)

        def empty_foundation(i: int) -> str:
            return paint(f"[ {SUITS[i]} ]", Card(SUITS[i], 1))

        top_row = [
            "[###]" if len(self.stock) else "[   ]",
            cell(self.waste.peek()),
            " " * 5,
        ]
        for i, f in enumerate(FOUNDATION_NAMES):
            card = self.piles[f].peek()
            top_row.append(cell(card) if card else empty_foundation(i))

        header1 = ["Stock", "Waste", "", "F1", "F2", "F3", "F4"]
        header2 = list(TABLEAU_NAMES)

        columns = [self.piles[t].bottom_to_top() for t in TABLEAU_NAMES]
        height = max(len(c) for c in columns)

        elapsed = int(self.elapsed())
        lines = [
            f"Score: {self.score}   Moves: {self.moves}   "
            f"Time: {elapsed // 60:02d}:{elapsed % 60:02d}   "
            f"Draw: {self.draw_count}   Stock left: {len(self.stock)}",
            "",
            " ".join(h.ljust(5) for h in header1),
            " ".join(top_row),
            "",
            " ".join(h.ljust(5) for h in header2),
        ]
        for r in range(height):
            parts = []
            for col in columns:
                parts.append(cell(col[r]) if r < len(col) else " " * 5)
            lines.append(" ".join(parts).rstrip())
        return "\n".join(lines)


# ======================================================================
# PHASE 4 - Text user interface
# ======================================================================
HELP = """
Commands
  d                 draw from the stock (turns the waste over when stock is empty)
  m SRC DST [N]     move N cards (default 1) from SRC to DST
                      SRC: W (waste), T1..T7, F1..F4
                      DST: T1..T7, F1..F4, or just F (picks the right foundation)
                      examples:  m W T3    m T2 F    m T1 T4 3
  u / r             undo / redo
  h                 hint (a legal move you could make)
  n                 new game
  help              show this help
  q                 quit
Foundations: F1=♠  F2=♥  F3=♦  F4=♣. Build each from Ace up to King.
""".strip()


def fmt_hint(h: tuple[str, str, int]) -> str:
    src, dst, n = h
    return f"m {src} {dst}" + (f" {n}" if n > 1 else "")


def main(argv: list[str]) -> None:
    draw_count = 3 if "--draw3" in argv else 1
    color = "--no-color" not in argv
    seed = None
    if "--seed" in argv:
        seed = int(argv[argv.index("--seed") + 1])

    try:                                   # make ♠♥♦♣ print on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    game = Game(draw_count=draw_count, seed=seed)
    print("KLONDIKE SOLITAIRE  -  type 'help' for commands\n")

    while True:
        print(game.render(color))
        if game.is_won():
            t = int(game.elapsed())
            print(f"\n*** YOU WON! ***  Score {game.score}, {game.moves} moves, {t // 60:02d}:{t % 60:02d}")
            if input("Play again? (y/n) ").strip().lower() == "y":
                game = Game(draw_count=draw_count)
                continue
            break

        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        print()

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd == "help":
            print(HELP)
        elif cmd == "d":
            print(game.draw()[1])
        elif cmd == "u":
            print(game.undo()[1])
        elif cmd == "r":
            print(game.redo()[1])
        elif cmd == "n":
            game = Game(draw_count=draw_count)
        elif cmd == "h":
            hints = game.hints()
            if hints:
                print("Hint:", fmt_hint(hints[0]))
            elif len(game.stock) or len(game.waste):
                print("Hint: no moves on the board - try drawing (d).")
            else:
                print("No moves left. Try undo (u) or a new game (n).")
        elif cmd == "m" and len(parts) in (3, 4):
            try:
                n = int(parts[3]) if len(parts) == 4 else 1
            except ValueError:
                print("N must be a number, e.g. m T1 T4 3")
                continue
            print(game.move(parts[1], parts[2], n)[1])
        else:
            print("Unknown command. Type 'help'.")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
