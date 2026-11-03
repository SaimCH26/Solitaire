"""Tests mirroring the project's Validation and Testing checklist.

Run:  python -m unittest -v test_solitaire
"""
import random
import unittest

from solitaire import (
    Card, Deck, Game, SUITS, TABLEAU_NAMES, FOUNDATION_NAMES, TableauPile,
)


def blank_game(draw_count=1) -> Game:
    """A game with every pile emptied, so tests can set up exact positions."""
    g = Game(draw_count=draw_count, seed=0)
    for name, pile in g.piles.items():
        if isinstance(pile, TableauPile):
            g.piles[name] = TableauPile()
        else:
            while len(pile):
                pile.pop()
    while len(g.stock):
        g.stock.dequeue()
    g.location.clear()
    return g


def put(g, name, *cards):
    """Place cards bottom -> top. Card spec: (suit, rank, face_up)."""
    for suit, rank, up in cards:
        c = Card(suit, rank, up)
        g.piles[name].push(c)
        g.location[c.key] = name


class TestPhase1CardAndDeck(unittest.TestCase):
    def test_deck_has_52_unique_cards(self):
        d = Deck()
        self.assertEqual(len(d), 52)
        self.assertEqual(len({c.key for c in d.cards}), 52)

    def test_shuffle_changes_order_but_keeps_cards(self):
        d = Deck()
        before = [c.key for c in d.cards]
        d.shuffle(random.Random(1))
        after = [c.key for c in d.cards]
        self.assertNotEqual(before, after)
        self.assertEqual(sorted(before), sorted(after))

    def test_same_seed_same_shuffle(self):
        a, b = Deck(), Deck()
        a.shuffle(random.Random(7))
        b.shuffle(random.Random(7))
        self.assertEqual([c.key for c in a.cards], [c.key for c in b.cards])


class TestInitialDeal(unittest.TestCase):
    def test_layout(self):
        g = Game(seed=3)
        for i, name in enumerate(TABLEAU_NAMES):
            pile = g.piles[name]
            self.assertEqual(len(pile), i + 1)
            self.assertEqual(pile.face_up_count(), 1)           # only the top is up
            self.assertTrue(pile.peek().face_up)
        self.assertEqual(len(g.stock), 24)
        self.assertTrue(all(not c.face_up for c in g.stock.to_list()))
        self.assertEqual(len(g.location), 52)


class TestLegalMoves(unittest.TestCase):
    def test_alternating_color_and_rank(self):
        g = blank_game()
        put(g, "T1", ("♠", 7, True))
        put(g, "T2", ("♥", 6, True))     # red 6 on black 7: legal
        put(g, "T3", ("♦", 6, True))
        ok, _ = g.move("T2", "T1")
        self.assertTrue(ok)
        g2 = blank_game()
        put(g2, "T1", ("♥", 7, True))
        put(g2, "T2", ("♦", 6, True))    # red on red: illegal
        self.assertFalse(g2.move("T2", "T1")[0])
        g3 = blank_game()
        put(g3, "T1", ("♠", 7, True))
        put(g3, "T2", ("♥", 5, True))    # wrong rank
        self.assertFalse(g3.move("T2", "T1")[0])

    def test_only_kings_on_empty_columns(self):
        g = blank_game()
        put(g, "T1", ("♠", 9, False), ("♠", 13, True))
        put(g, "T2", ("♥", 5, True))
        self.assertTrue(g.move("T1", "T3")[0])
        self.assertFalse(g.move("T2", "T4")[0])

    def test_foundation_rules(self):
        g = blank_game()
        put(g, "T1", ("♠", 2, True))
        put(g, "T2", ("♠", 1, True))
        put(g, "T3", ("♥", 1, True))
        self.assertFalse(g.move("T1", "F")[0])        # 2 before Ace
        self.assertTrue(g.move("T2", "F")[0])         # Ace of spades
        self.assertTrue(g.move("T1", "F")[0])         # then the 2
        self.assertFalse(g.move("T3", "F1")[0])       # wrong suit pile
        self.assertTrue(g.move("T3", "F2")[0])

    def test_cannot_move_face_down_cards(self):
        g = blank_game()
        put(g, "T1", ("♠", 8, False), ("♥", 5, True))
        put(g, "T2", ("♠", 6, True))
        self.assertFalse(g.move("T1", "T2", 2)[0])    # only 1 face-up card

    def test_multi_card_run_moves_in_order(self):
        g = blank_game()
        put(g, "T1", ("♠", 9, True), ("♥", 8, True), ("♣", 7, True))
        put(g, "T2", ("♥", 10, True))
        ok, msg = g.move("T1", "T2", 3)
        self.assertTrue(ok, msg)
        self.assertEqual([c.rank for c in g.piles["T2"].bottom_to_top()], [10, 9, 8, 7])
        self.assertEqual(len(g.piles["T1"]), 0)

    def test_invalid_run_rejected(self):
        g = blank_game()
        put(g, "T1", ("♠", 9, True), ("♣", 8, True))   # same color: not a run
        put(g, "T2", ("♥", 10, True))
        self.assertFalse(g.move("T1", "T2", 2)[0])


class TestCardFlipping(unittest.TestCase):
    def test_card_flips_after_move_and_undo_unflips(self):
        g = blank_game()
        put(g, "T1", ("♠", 9, False), ("♠", 1, True))
        before = g.snapshot()
        ok, _ = g.move("T1", "F")
        self.assertTrue(ok)
        self.assertTrue(g.piles["T1"].peek().face_up)
        g.undo()
        # The Ace is back on top (face up); the 9 underneath must be face down again.
        bottom, top = g.piles["T1"].bottom_to_top()
        self.assertFalse(bottom.face_up)
        self.assertTrue(top.face_up)
        self.assertEqual(g.snapshot(), before)


class TestStock(unittest.TestCase):
    def test_draw_one_and_three(self):
        g = Game(draw_count=1, seed=1)
        g.draw()
        self.assertEqual((len(g.stock), len(g.waste)), (23, 1))
        g3 = Game(draw_count=3, seed=1)
        g3.draw()
        self.assertEqual((len(g3.stock), len(g3.waste)), (21, 3))

    def test_fifo_and_recycle_order(self):
        g = Game(seed=1)
        order = [c.key for c in g.stock.to_list()]
        drawn = []
        for _ in range(24):
            g.draw()
            drawn.append(g.waste.peek().key)
        self.assertEqual(drawn, order)                # first in = first out
        g.draw()                                      # stock empty -> recycle
        self.assertEqual(len(g.stock), 24)
        self.assertEqual([c.key for c in g.stock.to_list()], order)
        g.draw()
        self.assertEqual(g.waste.peek().key, order[0])

    def test_nothing_to_draw(self):
        g = blank_game()
        self.assertFalse(g.draw()[0])


class TestWin(unittest.TestCase):
    def test_new_game_not_won(self):
        self.assertFalse(Game(seed=1).is_won())

    def test_win_detected(self):
        g = blank_game()
        for i, f in enumerate(FOUNDATION_NAMES):
            for rank in range(1, 13):
                c = Card(SUITS[i], rank, True)
                g.piles[f].push(c)
                g.location[c.key] = f
        put(g, "T1", ("♠", 13, True))
        put(g, "T2", ("♥", 13, True))
        put(g, "T3", ("♦", 13, True))
        put(g, "T4", ("♣", 13, True))
        self.assertFalse(g.is_won())
        for t in ("T1", "T2", "T3", "T4"):
            self.assertTrue(g.move(t, "F")[0])
        self.assertTrue(g.is_won())
        self.assertIsNotNone(g.finished_at)


class TestUndoRedo(unittest.TestCase):
    def test_redo_cleared_by_new_action(self):
        g = Game(seed=2)
        g.draw()
        g.undo()
        g.draw()
        self.assertFalse(g.redo()[0])

    def test_random_playout_undo_and_redo_are_exact(self):
        for seed in range(5):
            for draw_count in (1, 3):
                g = Game(draw_count=draw_count, seed=seed)
                rng = random.Random(seed)
                start = g.snapshot()
                piles = ["W"] + list(TABLEAU_NAMES) + list(FOUNDATION_NAMES)
                for _ in range(400):
                    if rng.random() < 0.3:
                        g.draw()
                    else:
                        hints = g.hints()
                        if hints and rng.random() < 0.8:
                            src, dst, n = rng.choice(hints)
                            g.move(src, dst, n)
                        else:
                            g.move(rng.choice(piles), rng.choice(piles + ["F"]), rng.randint(1, 3))
                end = g.snapshot()
                self._check_locations(g)
                while g.undo()[0]:
                    pass
                self.assertEqual(g.snapshot(), start)
                self._check_locations(g)
                while g.redo()[0]:
                    pass
                self.assertEqual(g.snapshot(), end)
                self._check_locations(g)

    def _check_locations(self, g):
        total = 0
        for name, pile in g.piles.items():
            cards = pile.bottom_to_top()
            total += len(cards)
            for c in cards:
                self.assertEqual(g.where(c), name)
        for c in g.stock.to_list():
            total += 1
            self.assertEqual(g.where(c), "S")
        self.assertEqual(total, 52)


class TestHints(unittest.TestCase):
    def test_hint_finds_foundation_move_first(self):
        g = blank_game()
        put(g, "T1", ("♠", 1, True))
        put(g, "T2", ("♥", 7, True))
        put(g, "T3", ("♣", 6, True))
        h = g.hints()
        self.assertEqual(h[0], ("T1", "F1", 1))
        self.assertIn(("T3", "T2", 1), h)


if __name__ == "__main__":
    unittest.main()
