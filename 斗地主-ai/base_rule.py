import random
from typing import List, Dict, Optional, Tuple,Any
from collections import defaultdict
class Card:
    def __init__(self, suit, rank):
        self.suit = suit  # 花色: ♠, ♥, ♦, ♣, 小王, 大王
        self.rank = rank  # 牌面: 3-10, J, Q, K, A, 2, 小王, 大王

    def __str__(self):
        return f"{self.suit}{self.rank}"

    def __repr__(self):
        return self.__str__()

    def get_value(self):
        """获取牌的数值，用于比较大小"""
        rank_values = {
            '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13,
            'A': 14, '2': 15, '小王': 16, '大王': 17
        }
        return rank_values.get(self.rank, 0)


class Deck:
    """牌堆类，包含54张斗地主牌"""

    def __init__(self):
        self.cards = []
        self._create_deck()

    def _create_deck(self):
        """创建标准的54张斗地主牌"""
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']

        # 添加普通牌
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))

        # 添加大小王
        self.cards.append(Card('', '小王'))
        self.cards.append(Card('', '大王'))

    def shuffle(self):
        """洗牌"""
        import random
        random.shuffle(self.cards)

    def deal(self, num_players=3):
        """发牌给玩家，返回一个包含各玩家手牌的列表和底牌"""
        self.shuffle()

        # 斗地主发牌规则：3个玩家，每人17张，留3张底牌
        hands = [[] for _ in range(num_players)]

        for i in range(51):  # 前51张牌轮流发给玩家
            player_idx = i % num_players
            hands[player_idx].append(self.cards[i])

        # 剩下的3张作为底牌
        bottom_cards = self.cards[51:]

        # 按牌值排序（可选）
        for hand in hands:
            hand.sort(key=lambda card: card.get_value())
        bottom_cards.sort(key=lambda card: card.get_value())

        return hands, bottom_cards

    def __str__(self):
        return str(self.cards)


class Player:
    """玩家基类"""

    def __init__(self, name: str):
        self.name = name
        self.hand: List[Card] = []  # 手牌
        self.role: Optional[str] = None  # 'landlord'地主或'peasant'农民
        self.bid_score: int = 0  # 叫分(0表示不叫，1/2/3分)

    def receive_cards(self, cards: List[Card]):
        """接收手牌"""
        self.hand.extend(cards)
        self.sort_hand()

    def make_bid(self, current_max_bid: int, game_state: Dict[str, Any]) -> int:
        """
        决定叫地主分数
        :param current_max_bid: 当前最高叫分(0-3)
        :param game_state: 游戏状态信息
        :return: 叫分(必须比current_max_bid高，0表示不叫)
        """
        raise NotImplementedError("子类必须实现make_bid方法")

    def become_landlord(self, bottom_cards: List[Card]):
        """
        成为地主，接收底牌
        """
        self.role = 'landlord'
        self.receive_cards(bottom_cards)
        print(f"{self.name} 成为地主，获得底牌: {bottom_cards}")

    def update_role(self, is_landlord: bool):
        """
        更新角色(地主或农民)
        """
        self.role = 'landlord' if is_landlord else 'peasant'
    def sort_hand(self):
        """按牌力排序手牌"""
        self.hand.sort(key=lambda card: card.get_value())

    def play_cards(self, game_state: Dict[str, Any]) -> Optional[List[Card]]:
        """出牌逻辑（基类方法，子类应重写）"""
        raise NotImplementedError("子类必须实现play_cards方法")

    def remove_cards_from_hand(self, cards_to_remove: List[Card]) -> bool:
        """从手牌中移除已出的牌"""
        original_count = len(self.hand)

        # 创建临时副本避免修改原列表
        temp_hand = self.hand.copy()

        for card in cards_to_remove:
            try:
                temp_hand.remove(card)
            except ValueError:
                # 如果牌不在手牌中
                return False

        # 只有所有牌都成功移除才更新手牌
        if len(temp_hand) == original_count - len(cards_to_remove):
            self.hand = temp_hand
            return True
        return False

    def can_play(self, cards: List[Card], last_move: Optional[List[Card]]) -> bool:
        """检查是否可以出这些牌"""
        if not cards:  # 不出牌是允许的
            return True

        # 检查所有牌是否都在手牌中
        if not all(card in self.hand for card in cards):
            return False

        # 如果没有上家出牌，任何合法牌型都可以出
        if not last_move:
            return self._is_valid_card_combination(cards)

        # 检查牌型匹配且大于上家
        return (self._is_same_combination_type(cards, last_move) and
                self._is_stronger_combination(cards, last_move))

    def _is_valid_card_combination(self, cards: List[Card]) -> bool:
        """检查是否是合法牌型"""
        # 这里简化实现，实际应该实现所有斗地主牌型判断
        length = len(cards)

        if length == 1:  # 单张
            return True
        elif length == 2:  # 对子或王炸
            return cards[0].rank == cards[1].rank or \
                {cards[0].rank, cards[1].rank} == {'小王', '大王'}

        return False

    def _is_same_combination_type(self, cards: List[Card], last_move: List[Card]) -> bool:
        """检查是否是相同牌型"""
        # 获取双方牌型
        cards_type = self._get_combination_type(cards)
        last_move_type = self._get_combination_type(last_move)

        # 特殊牌型：王炸可以压任何牌
        if cards_type == "rocket":
            return True
        if last_move_type == "rocket":
            return False

        # 炸弹可以压非炸弹牌型
        if cards_type == "bomb" and last_move_type != "bomb":
            return True
        if last_move_type == "bomb" and cards_type != "bomb":
            return False

        # 其他情况需要牌型完全相同
        return cards_type == last_move_type

    def _is_stronger_combination(self, cards: List[Card], last_move: List[Card]) -> bool:
        """检查是否比上家的牌大"""
        cards_type = self._get_combination_type(cards)
        last_move_type = self._get_combination_type(last_move)

        # 王炸最大
        if cards_type == "rocket":
            return True
        if last_move_type == "rocket":
            return False

        # 炸弹比较
        if cards_type == "bomb" and last_move_type == "bomb":
            return cards[0].get_value() > last_move[0].get_value()

        # 相同牌型比较
        if cards_type == last_move_type:
            return self._compare_same_type(cards, last_move, cards_type)

        return False

    def _get_combination_type(self, cards: List[Card]) -> str:
        """获取牌型"""
        if not cards:
            return "pass"

        length = len(cards)
        rank_count = self._count_ranks(cards)
        values = [card.get_value() for card in cards]

        # 王炸
        if length == 2 and {cards[0].rank, cards[1].rank} == {'小王', '大王'}:
            return "rocket"

        # 炸弹
        if length == 4 and len(rank_count) == 1:
            return "bomb"

        # 单张
        if length == 1:
            return "single"

        # 对子
        if length == 2 and len(rank_count) == 1:
            return "pair"

        # 三张
        if length == 3 and len(rank_count) == 1:
            return "triplet"

        # 三带一
        if length == 4 and any(count == 3 for count in rank_count.values()):
            return "triplet_with_single"

        # 三带对
        if length == 5 and any(count == 3 for count in rank_count.values()) and \
                any(count == 2 for count in rank_count.values()):
            return "triplet_with_pair"

        # 连对 (至少3个连续对子)
        if length >= 6 and length % 2 == 0 and \
                self._is_consecutive_pairs(cards):
            return "consecutive_pairs"

        # 顺子 (至少5张连续单牌)
        if length >= 5 and self._is_straight(values):
            return "straight"

        # 飞机 (多个连续三张)
        if length >= 6 and self._is_airplane(cards):
            return "airplane"

        # 其他未识别牌型
        return "invalid"

    def _compare_same_type(self, cards: List[Card], last_move: List[Card], cards_type: str) -> bool:
        """比较相同牌型的大小"""
        if cards_type in ["single", "pair", "triplet", "bomb"]:
            return cards[0].get_value() > last_move[0].get_value()

        elif cards_type == "triplet_with_single":
            # 比较三张部分的牌
            cards_triplet = self._get_triplet_part(cards)
            last_triplet = self._get_triplet_part(last_move)
            return cards_triplet[0].get_value() > last_triplet[0].get_value()

        elif cards_type == "triplet_with_pair":
            # 比较三张部分的牌
            cards_triplet = self._get_triplet_part(cards)
            last_triplet = self._get_triplet_part(last_move)
            return cards_triplet[0].get_value() > last_triplet[0].get_value()

        elif cards_type == "consecutive_pairs":
            # 比较最小的对子
            return min(card.get_value() for card in cards) > \
                min(card.get_value() for card in last_move)

        elif cards_type == "straight":
            # 比较最大的牌
            return max(card.get_value() for card in cards) > \
                max(card.get_value() for card in last_move)

        elif cards_type == "airplane":
            # 比较最小的三张牌
            triplets = self._get_all_triplets(cards)
            last_triplets = self._get_all_triplets(last_move)
            return min(t[0].get_value() for t in triplets) > \
                min(t[0].get_value() for t in last_triplets)

        return False

    # ========== 辅助方法 ==========

    def _count_ranks(self, cards: List[Card]) -> Dict[str, int]:
        """统计每种牌面的数量"""
        rank_count = defaultdict(int)
        for card in cards:
            rank_count[card.rank] += 1
        return rank_count

    def _is_straight(self, values: List[int]) -> bool:
        """检查是否是连续的牌(顺子)"""
        if len(values) < 5:
            return False

        # 去除重复并排序
        unique_sorted = sorted(set(values))

        # 检查是否连续
        for i in range(1, len(unique_sorted)):
            if unique_sorted[i] - unique_sorted[i - 1] != 1:
                return False
        return True

    def _is_consecutive_pairs(self, cards: List[Card]) -> bool:
        """检查是否是连续的对子"""
        rank_count = self._count_ranks(cards)

        # 检查是否都是对子
        if any(count != 2 for count in rank_count.values()):
            return False

        # 获取所有牌面值并排序
        values = sorted({card.get_value() for card in cards})

        # 检查是否连续
        for i in range(1, len(values)):
            if values[i] - values[i - 1] != 1:
                return False
        return True

    def _get_triplet_part(self, cards: List[Card]) -> List[Card]:
        """从三带牌中获取三张部分"""
        rank_count = self._count_ranks(cards)
        triplet_rank = next(rank for rank, count in rank_count.items() if count == 3)
        return [card for card in cards if card.rank == triplet_rank]

    def _is_airplane(self, cards: List[Card]) -> bool:
        """检查是否是飞机(多个连续三张)"""
        rank_count = self._count_ranks(cards)
        triplet_ranks = [rank for rank, count in rank_count.items() if count == 3]

        if len(triplet_ranks) < 2:
            return False

        # 获取三张牌的牌面值并排序
        values = sorted([Card('', rank).get_value() for rank in triplet_ranks])

        # 检查是否连续
        for i in range(1, len(values)):
            if values[i] - values[i - 1] != 1:
                return False

        # 检查带牌是否合理
        total_cards = len(cards)
        num_triplets = len(triplet_ranks)
        expected_length = num_triplets * 3

        # 纯飞机
        if total_cards == expected_length:
            return True

        # 飞机带单张
        if total_cards == num_triplets * 4:
            single_count = sum(1 for count in rank_count.values() if count == 1)
            return single_count == num_triplets

        # 飞机带对子
        if total_cards == num_triplets * 5:
            pair_count = sum(1 for count in rank_count.values() if count == 2)
            return pair_count == num_triplets

        return False

    def _get_all_triplets(self, cards: List[Card]) -> List[List[Card]]:
        """获取所有的三张牌组合"""
        rank_count = self._count_ranks(cards)
        triplets = []
        for rank, count in rank_count.items():
            if count == 3:
                triplet = [card for card in cards if card.rank == rank]
                triplets.append(triplet)
        return triplets

    def __str__(self):
        return f"{self.name}({self.role})" if self.role else self.name


class HumanPlayer(Player):
    """人类玩家"""

    def make_bid(self, current_max_bid: int, game_state: Dict[str, Any]) -> int:
        print(f"\n{self.name}的手牌: {self.hand}")
        print(f"当前最高叫分: {current_max_bid}分")
        print("请叫分(输入0不叫，1/2/3分，必须比当前最高分高):")

        while True:
            try:
                bid = int(input().strip())
                if bid == 0:
                    return 0
                if bid not in {1, 2, 3}:
                    print("叫分必须是0-3的整数")
                    continue
                if bid <= current_max_bid:
                    print(f"叫分必须比当前最高分{current_max_bid}高")
                    continue
                return bid
            except ValueError:
                print("请输入有效的数字(0-3)")
    def play_cards(self, game_state: Dict[str, Any]) -> Optional[List[Card]]:
        print(f"\n{self.name}的手牌: {self.hand}")
        print("请输入要出的牌(用空格分隔，例如'♥A ♠2'，不出请直接回车或输入空格):")

        while True:
            try:
                input_str = input().strip()

                # 明确处理空输入或纯空格的情况
                if not input_str or input_str.isspace():
                    return None  # 表示不出牌

                # 解析输入的牌
                card_strs = [s for s in input_str.split() if s]  # 过滤掉空字符串
                if not card_strs:  # 再次检查（理论上不会执行到这里）
                    return None

                selected_cards = []
                invalid_cards = []

                # 在手牌中查找对应的牌
                for card_str in card_strs:
                    found = False
                    for card in self.hand:
                        if str(card) == card_str:
                            selected_cards.append(card)
                            found = True
                            break
                    if not found:
                        invalid_cards.append(card_str)

                # 检查是否有无效的牌
                if invalid_cards:
                    raise ValueError(f"无效的牌: {', '.join(invalid_cards)}")

                # 验证出牌是否合法
                last_move = game_state.get('last_move')
                if self.can_play(selected_cards, last_move):
                    return selected_cards
                else:
                    print("出牌不合法，请重新输入:")
            except Exception as e:
                print(f"输入错误: {e}，请重新输入:")


class AIPlayer(Player):
    """AI玩家"""

    def __init__(self, name: str, difficulty: str = 'medium'):
        super().__init__(name)
        self.difficulty = difficulty

    def play_cards(self, game_state: Dict[str, Any]) -> Optional[List[Card]]:
        last_move = game_state.get('last_move')
        legal_moves = self._get_legal_moves(last_move)

        if not legal_moves:  # 没有合法出牌
            return None

        # 根据难度选择策略
        if self.difficulty == 'easy':
            return self._easy_strategy(legal_moves)
        elif self.difficulty == 'medium':
            return self._medium_strategy(legal_moves, game_state)
        else:  # hard
            return self._hard_strategy(legal_moves, game_state)

    def _get_legal_moves(self, last_move: Optional[List[Card]]) -> List[List[Card]]:
        """获取所有合法出牌组合"""
        # 这里简化实现，实际应该生成所有合法牌型组合
        legal_moves = []

        if not last_move:  # 没有上家出牌，可以出任意合法牌型
            # 单张
            for card in self.hand:
                legal_moves.append([card])

            # 对子
            rank_count = {}
            for card in self.hand:
                rank_count[card.rank] = rank_count.get(card.rank, 0) + 1

            for rank, count in rank_count.items():
                if count >= 2:
                    pair = [card for card in self.hand if card.rank == rank][:2]
                    legal_moves.append(pair)

            # 王炸
            if {'小王', '大王'}.issubset({card.rank for card in self.hand}):
                legal_moves.append(
                    [card for card in self.hand if card.rank in {'小王', '大王'}]
                )
        else:
            # 有上家出牌，需要找能压过的相同牌型
            # 简化实现：只找相同数量且更大的单牌或对子
            target_len = len(last_move)
            target_value = max(card.get_value() for card in last_move)

            if target_len == 1:  # 上家出单张
                for card in self.hand:
                    if card.get_value() > target_value:
                        legal_moves.append([card])
            elif target_len == 2:  # 上家出对子
                rank_count = {}
                for card in self.hand:
                    if card.get_value() > target_value:
                        rank_count[card.rank] = rank_count.get(card.rank, 0) + 1

                for rank, count in rank_count.items():
                    if count >= 2:
                        pair = [card for card in self.hand if card.rank == rank][:2]
                        legal_moves.append(pair)

        return legal_moves

    def _easy_strategy(self, legal_moves: List[List[Card]]) -> List[Card]:
        """简单策略：随机出牌"""
        return random.choice(legal_moves + [[]])  # 包含不出牌的选择

    def _medium_strategy(self, legal_moves: List[List[Card]], game_state: Dict[str, Any]) -> List[Card]:
        """中等策略：基本逻辑"""
        if not legal_moves:
            return None

        # 如果是地主且是第一个出牌，出小牌
        if self.role == 'landlord' and not game_state.get('last_move'):
            return min(legal_moves, key=lambda move: max(card.get_value() for card in move))

        # 如果手牌少，尽量出大牌
        if len(self.hand) <= 3:
            return max(legal_moves, key=lambda move: max(card.get_value() for card in move))

        # 默认出最小的能压过的牌
        return min(legal_moves, key=lambda move: max(card.get_value() for card in move))

    def _hard_strategy(self, legal_moves: List[List[Card]], game_state: Dict[str, Any]) -> List[Card]:
        """困难策略：更复杂的逻辑"""
        # 这里可以添加更复杂的AI逻辑，例如：
        # - 记牌
        # - 分析对手可能的手牌
        # - 保留关键牌
        # - 拆牌策略等

        # 暂时使用中等策略
        return self._medium_strategy(legal_moves, game_state)

    def make_bid(self, current_max_bid: int, game_state: Dict[str, Any]) -> int:
        # 根据手牌质量决定叫分
        hand_quality = self._evaluate_hand_quality()

        # 叫分策略基于手牌质量和当前最高分
        if self.difficulty == 'easy':
            return self._easy_bid_strategy(hand_quality, current_max_bid)
        elif self.difficulty == 'medium':
            return self._medium_bid_strategy(hand_quality, current_max_bid)
        else:
            return self._hard_bid_strategy(hand_quality, current_max_bid, game_state)

    def _evaluate_hand_quality(self) -> float:
        """评估手牌质量(0-1之间的值)"""
        # 简单实现：基于高牌数量
        high_cards = sum(1 for card in self.hand if card.get_value() >= 10)
        return high_cards / len(self.hand) if self.hand else 0

    def _easy_bid_strategy(self, hand_quality: float, current_max_bid: int) -> int:
        """简单AI叫分策略"""
        if random.random() < 0.3:  # 30%概率叫分
            return min(3, current_max_bid + 1)
        return 0

    def _medium_bid_strategy(self, hand_quality: float, current_max_bid: int) -> int:
        """中等AI叫分策略"""
        if hand_quality > 0.6 and current_max_bid < 3:
            return current_max_bid + 1
        elif hand_quality > 0.4 and current_max_bid < 2:
            return current_max_bid + 1
        return 0

    def _hard_bid_strategy(self, hand_quality: float, current_max_bid: int,
                           game_state: Dict[str, Any]) -> int:
        """困难AI叫分策略"""
        # 考虑更多因素：炸弹数量、牌型完整性等
        bomb_count = self._count_bombs()
        if bomb_count >= 2 and current_max_bid < 3:
            return 3
        elif hand_quality > 0.7 and current_max_bid < 2:
            return 2
        elif hand_quality > 0.5 and current_max_bid < 1:
            return 1
        return 0

    def _count_bombs(self) -> int:
        """计算手牌中的炸弹数量"""
        rank_count = {}
        for card in self.hand:
            rank_count[card.rank] = rank_count.get(card.rank, 0) + 1
        return sum(1 for count in rank_count.values() if count >= 4)



class LandlordGame:
    """斗地主游戏主类"""

    def __init__(self, players: List[Player]):
        self.players = players
        self.deck = Deck()
        self.bottom_cards: List[Card] = []
        self.current_player_idx = 0  # 当前出牌玩家索引
        self.last_move: Optional[List[Card]] = None  # 上一手牌
        self.last_move_player: Optional[Player] = None  # 上一手牌玩家
        self.game_over = False
        self.winner: Optional[Player] = None
        self.landlord: Optional[Player] = None

    def start_game(self):
        """开始游戏主流程"""
        print("=== 游戏开始 ===")

        # 1. 洗牌和发牌
        self._deal_cards()

        # 2. 叫地主流程
        self._bid_for_landlord()

        # 3. 确定地主先出牌
        self.current_player_idx = self.players.index(self.landlord)
        print(f"\n=== 地主 {self.landlord.name} 先出牌 ===")

        # 4. 游戏主循环
        self._main_game_loop()

        # 5. 游戏结束处理
        self._end_game()

    def _deal_cards(self):
        """发牌"""
        self.deck = Deck()
        hands, self.bottom_cards = self.deck.deal()

        for i, player in enumerate(self.players):
            player.receive_cards(hands[i])
            print(f"{player.name} 获得 {len(hands[i])}张牌")

        print(f"底牌: {self.bottom_cards}")

    def _bid_for_landlord(self):
        """叫地主流程"""
        current_bid = 0
        bid_winner = None

        # 随机决定第一个叫地主的玩家
        first_bidder_idx = random.randint(0, 2)
        bid_order = [
            first_bidder_idx,
            (first_bidder_idx + 1) % 3,
            (first_bidder_idx + 2) % 3
        ]

        print("\n=== 叫地主环节 ===")
        print(f"{self.players[first_bidder_idx].name} 首先叫分")

        for i in bid_order:
            player = self.players[i]
            bid = player.make_bid(current_bid, {
                'current_bid': current_bid,
                'bid_round': 0
            })

            if bid > current_bid:
                current_bid = bid
                bid_winner = player
                print(f"{player.name} 叫 {bid}分")

            if current_bid == 3:  # 叫到最高分，提前结束
                break

        # 确定地主
        if bid_winner:
            self.landlord = bid_winner
            bid_winner.become_landlord(self.bottom_cards)
            for player in self.players:
                if player != bid_winner:
                    player.update_role(False)
            print(f"\n=== {bid_winner.name} 成为地主 ===")
        else:
            # 如果没人叫分，默认第一个叫分的玩家当地主
            self.landlord = self.players[first_bidder_idx]
            self.landlord.become_landlord(self.bottom_cards)
            for player in self.players:
                if player != self.landlord:
                    player.update_role(False)
            print("\n=== 无人叫分，默认第一个玩家成为地主 ===")

        # 显示所有玩家角色
        for player in self.players:
            print(f"{player.name}: {player.role}")

    def _main_game_loop(self):
        """游戏主循环"""
        round_count = 1

        while not self.game_over:
            print(f"\n=== 第{round_count}轮出牌 ===")

            # 获取当前玩家
            current_player = self.players[self.current_player_idx]

            # 显示游戏状态
            self._display_game_state(current_player)

            # 玩家出牌
            played_cards = current_player.play_cards(self._get_game_state())

            if played_cards:
                # 验证出牌合法性
                if not current_player.can_play(played_cards, self.last_move):
                    print("无效出牌！请重新选择")
                    continue

                # 从手牌移除已出的牌
                current_player.remove_cards_from_hand(played_cards)

                # 更新游戏状态
                self.last_move = played_cards
                self.last_move_player = current_player
                print(f"{current_player.name} 出牌: {played_cards}")

                # 检查是否出完牌
                if not current_player.hand:
                    self.game_over = True
                    self.winner = current_player
                    break
            else:
                print(f"{current_player.name} 选择不出")

            # 转到下一个玩家
            self.current_player_idx = (self.current_player_idx + 1) % 3

            # 如果一圈都选择不出，清空上家出牌
            if (self.last_move_player and
                    self.players[self.current_player_idx] == self.last_move_player):
                print("一轮结束，清空上家出牌")
                self.last_move = None
                self.last_move_player = None

            round_count += 1

    def _get_game_state(self) -> Dict[str, Any]:
        """获取当前游戏状态信息"""
        return {
            'last_move': self.last_move,
            'current_player': self.players[self.current_player_idx].name,
            'landlord': self.landlord.name if self.landlord else None,
            'players_hand_count': {
                player.name: len(player.hand) for player in self.players
            },
            'bottom_cards': self.bottom_cards
        }

    def _display_game_state(self, current_player: Player):
        """显示当前游戏状态"""
        print(f"\n当前玩家: {current_player.name}")
        print(f"上家出牌: {self.last_move if self.last_move else '无'}")

        # 显示各玩家剩余牌数
        for player in self.players:
            if player == current_player and isinstance(player, HumanPlayer):
                print(f"{player.name}的手牌({player.role}): {player.hand}")
            else:
                print(f"{player.name}({player.role})剩余牌数: {len(player.hand)}")

    def _end_game(self):
        """游戏结束处理"""
        print("\n=== 游戏结束 ===")

        if self.winner:
            print(f"{self.winner.name} 第一个出完牌，游戏胜利！")

            # 计算得分
            if self.winner.role == 'landlord':
                print("地主胜利！")
                # 这里可以添加具体得分计算
            else:
                print("农民胜利！")
                # 这里可以添加具体得分计算

        # 显示所有玩家剩余手牌
        for player in self.players:
            print(f"{player.name} 剩余手牌: {player.hand}")
if __name__ == '__main__':
    # 创建玩家
    # 创建游戏和玩家
    # 创建玩家
    players = [
        HumanPlayer("玩家1"),
        AIPlayer("电脑1", "medium"),
        AIPlayer("电脑2", "hard")
    ]

    # 创建并开始游戏
    game = LandlordGame(players)
    game.start_game()