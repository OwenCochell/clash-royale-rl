import numpy as np
import numpy.typing as npt
from queue import Queue

import pygame

import clash_royale.envs.game_engine.cards as cr


class Hand():
    def __init__(self, deck: list[type[cr.Card]]) -> None:
        self.deck = Queue(maxsize=8)
        for card in deck:
            self.deck.put(card, block=False)
        
        self.available = [self.deck.get(block=False) for i in range(4)]
        self.next = self.deck.get(block=False)

    def query(self, index: int) -> cr.Card:
        if index < 4:
            return self.available[index]
        else:
            return self.next
        
    def pop(self, index: int) -> None:
        self.deck.put(self.available[index], block=False)
        self.available[index] = self.next
        self.next = self.deck.get(block=False)

    def hand(self) -> Queue:
        return self.available

        

class Player():
    def __init__(self, deck: list[str]) -> None:
        self.elixir = 5.0

        deck = [cr.name_to_card[card_name] for card_name in deck]
        np.random.shuffle(deck)
        self.hand = Hand(deck)
        self.towers = 3
        self.king_tower_active = False

        #self.king_tower = cr.KingTower()
        #etc

    def get_pseudo_legal_cards(self) -> list[int]:
        hand = self.hand.hand()
        return [(card_idx, isinstance(hand[card_idx], cr.Spell)) for card_idx in range(len(hand)) if hand[card_idx] <= self.elixir]
    
    def play_card(self, index: int) -> None:
        elixir = self.hand.query(index).elixir
        self.hand.pop(index)
        self.elixir -= elixir

        assert index < 4 and self.elixir >= 0

    def step(self, elixir_rate: float, fps: int, frames: int=1) -> None:
        self.elixir += ((elixir_rate / fps) * frames)


class GameEngine():
    # note: mu-zero should never require a game to be copied
    def __init__(self, 
                 deck1: list[str] = ['knight' * 8], 
                 deck2: list[str] = ['knight' * 8], 
                 fps: int = 30,
                 resolution: npt.ArrayLike = [128, 128],
                 dimensions: npt.ArrayLike = [18, 32]) -> None:
        self.fps = fps
        self.resolution = resolution
        self.dimensions = dimensions

        self.images = []
        self.actions = []
        self.current_frame = 0

        self.state = 0
        self.placement_masks = np.ones(shape=(2, 18, 32, ), dtype=bool)
        self.elixir_rate = (1/2.8)

        self.player_1 = Player(deck1)
        self.player_2 = Player(deck2)

        self.game_over = False
        self.victor = None

        self.entity_list = []

    def image(self) -> np.ndarray:
        # display entities layered
        canvas = pygame.Surface(self.resolution)
        canvas.fill((255, 255, 255))

        # render entities
        self.entity_list.sort(key=lambda entity: entity.location[1])

        for entity in self.entity_list:
            entity.render(canvas)

        # render info

        return np.array(pygame.surfarray.pixels3d(canvas))

    def apply(self, action: tuple[int, int, int]) -> None:
        pass

    def update_state(self, frames:int) -> None:
        if self.state == 0 and self.current_frame >= self.fps * 120:
            self.state = 1
            self.elixir_rate = (2/2.8)
        if self.state == 1 and self.current_frame >= self.fps * 180:
            self.state = 2
            if self.player_1.towers != self.player_2.towers:
                self.game_over = True
                if self.player_1.towers > self.player_2.towers:
                    self.victor = 0
                else:
                    self.victor = 1

        if self.state == 2:
            if self.player_1.towers != self.player_2.towers:
                self.game_over = True
                if self.player_1.towers > self.player_2.towers:
                    self.victor = 0
                else:
                    self.victor = 1
            if self.current_frame >= self.fps * (180 + 60):
                self.elixir_rate = (3/2.8)
            if self.current_frame >= self.fps * (180 + 120):
                self.game_over = True
                #check draw overtime tower health condition for all towers
                self.victor = 0.5

    def update_troops(self, frames:int) -> None:
        pass

    def update_placement_masks(self) -> None:
        pass


    def step(self, frames:int = 1) -> None:
        if self.game_over:
            return
        
        # update elixir
        self.player_1.step(self.elixir_rate, self.fps, frames)
        self.player_2.step(self.elixir_rate, self.fps, frames)

        # update troop health, placement, etc
        self.current_frame += frames
        self.update_state(frames)
        self.update_troops(frames)

        self.update_placement_masks()
       

    def is_terminal(self) -> bool:
        return self.game_over

    def terminal_value(self) -> float:
        return self.victor

    def legal_actions(self, to_play: int) -> np.ndarray:
        actions = np.zeros(shape=(18, 32, 5), dtype=np.float64)
        actions[:,:,4] = 1 # no card is always legal
        if to_play == 0:
            hand = self.player_1.get_pseudo_legal_cards()
        else:
            hand = self.player_2.get_pseudo_legal_cards()

        for (card_index, is_spell) in hand:
            if is_spell:
                actions[:, :, card_index] = 1
            else:
                actions[self.placement_masks[to_play], card_index] = 1

        return actions


