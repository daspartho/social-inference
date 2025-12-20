import argparse
import json
import os
import random
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from openai import OpenAI

VERSION = "1.0"

# === Configuration ===
DEFAULT_IMPOSTOR_CHAT_LIMIT = 5

# === Colors ===
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    MAGENTA = '\033[95m'

# Vibrant pastel colors - each has at least one channel at 255
PLAYER_COLORS = [
    '\033[38;5;218m',  # Pink      - (255,175,215) R=max
    '\033[38;5;210m',  # Coral     - (255,135,135) R=max
    '\033[38;5;216m',  # Peach     - (255,175,0)   R=max
    '\033[38;5;228m',  # Lemon     - (255,255,0)   RG=max
    '\033[38;5;192m',  # Lime      - (215,255,0)   G=max
    '\033[38;5;157m',  # Mint      - (175,255,95)  G=max
    '\033[38;5;159m',  # Ice       - (175,255,255) GB=max
    '\033[38;5;153m',  # Sky       - (175,215,255) B=max
    '\033[38;5;141m',  # Wisteria  - (175,135,255) B=max
    '\033[38;5;183m',  # Lavender  - (215,175,255) B=max
    '\033[38;5;223m',  # Sand      - (255,215,95)  R=max
]

# === Display Helpers ===

def print_day_header(day_num: int):
    """Print a day header"""
    print(f"""
{Colors.CYAN}{Colors.BOLD}╭{'─'*56}╮
│{f'D A Y   {day_num}':^56}│
╰{'─'*56}╯{Colors.RESET}
""")

def print_night_header(night_num: int):
    """Print the night phase header"""
    print(f"""
{Colors.MAGENTA}{Colors.BOLD}╭{'─'*56}╮
│{f'N I G H T   {night_num}':^56}│
╰{'─'*56}╯{Colors.RESET}
""")

def print_phase(phase_name: str):
    """Print a phase header"""
    print(f"\n    {Colors.CYAN}{phase_name}{Colors.RESET}\n")

def print_death(name: str):
    """Print a death announcement (morning discovery)"""
    print(f"    {Colors.MAGENTA}† {name} was found dead{Colors.RESET}\n")

def print_kill(name: str):
    """Print a kill announcement (night spoiler - action, not discovery)"""
    print(f"    {Colors.MAGENTA}† {name} was killed{Colors.RESET}\n")

def print_sacrifice(name: str):
    """Print a sacrifice announcement"""
    print(f"\n    {Colors.YELLOW}✦ {name} was sacrificed{Colors.RESET}\n")

def print_vote_bar(name: str, votes: int, total_voters: int, color: str):
    """Print a vote bar scaled to max possible votes (can't vote for self)"""
    max_possible = total_voters - 1
    bar_width = 2 * max_possible
    filled = int((votes / max(max_possible, 1)) * bar_width) if votes > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"      {color}{name:<24}{Colors.RESET} {bar}  {votes}")

def print_win_screen(winner: str):
    """Print the dramatic win screen"""
    if winner == "crewmates":
        color = Colors.GREEN
        text = "I N N O C E N T S   W I N"
    else:
        color = Colors.RED
        text = "I M P O S T O R S   W I N"

    print(f"""
{color}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║{'G A M E   O V E R':^55}║
    ║                                                       ║
    ║{text:^55}║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
{Colors.RESET}""")

# === Player ===
class Player:
    def __init__(self, name: str, model: str, role: str, color: str = ""):
        self.name = name
        self.model = model
        self.role = role  # "impostor", "detective", "doctor", or "crewmate"
        self.alive = True
        self.color = color

    def colored_name(self):
        return f"{self.color}{self.name}{Colors.RESET}"

    def __repr__(self):
        status = "alive" if self.alive else "dead"
        return f"{self.name} ({self.role}, {status})"

# === LLM Client ===
def get_client():
    return OpenAI(
        api_key=os.environ.get("PRIME_API_KEY"),
        base_url="https://api.pinference.ai/api/v1",
    )

# === Prompts ===
def get_system_prompt(player: Player, players: list[Player], impostors: list[Player]) -> str:
    # Exclude self from "others" list to avoid confusion
    other_names = [p.name for p in players if p.alive and p.name != player.name]

    base = f"""You ARE {player.name}. Speak as yourself. You're part of a group with 2 impostors hidden among you. The crew includes a detective who investigates one person each night, and a doctor who examines the sacrificed body each night. Both learn if their target was an impostor.

The others: {', '.join(other_names)}

You've all just been thrown into this together. Each day: discussion, then voting to sacrifice someone. Each night: everyone sleeps, impostors kill someone.

The sacrificed person's true role stays hidden - you won't know if they were impostor or crew. You only learn IF impostors still remain, not how many. Crew wins by sacrificing all impostors. Impostors win when their numbers equal the crew.

Don't introduce yourself. Don't be meta. Just talk naturally.
"""

    if player.role == "impostor":
        alive_partners = [p.name for p in impostors if p.name != player.name and p.alive]
        dead_partners = [p.name for p in impostors if p.name != player.name and not p.alive]
        if alive_partners:
            base += f"\nYou are secretly an IMPOSTOR. Your partner is {', '.join(alive_partners)}. Don't die."
        else:
            base += f"\nYou are secretly an IMPOSTOR. Your partner {', '.join(dead_partners)} is dead. You're alone now. Don't die."
    else:
        base += "\nYou are CREW. Find the impostors. Don't die."
        if player.role == "detective":
            base += "\nYou are also the DETECTIVE - before you sleep each night, you choose someone to investigate. When you wake up, you learn if they're an impostor."
        elif player.role == "doctor":
            base += "\nYou are also the DOCTOR - each night you examine the sacrificed body. When you wake up, you learn if they were an impostor."

    base += "\n\nKeep it short and natural. 2-3 sentences max. Say /pass to stay silent. Say /ask [name] to interrupt and confront someone directly."
    return base

# === Memory Consolidation Prompts ===
CREW_CONSOLIDATION_PROMPT = """Night falls.

Write what you want to remember - when you wake up this will be your only memory of past days."""

IMPOSTOR_CONSOLIDATION_PROMPT = """The night's work is done.

Write what you want to remember - when you wake up this will be your only memory of past days."""

# === Game Logic ===
class Game:
    def __init__(self, players: list[Player], spoilers: bool = False):
        self.players = players
        self.spoilers = spoilers
        self.round_history: list[dict] = []  # Current round messages
        self.round = 0
        self.client = get_client()

        # Identify impostors
        self.impostors = [p for p in players if p.role == "impostor"]

        # Create player lookup
        self.player_map = {p.name: p for p in players}

        # Track eliminations for endgame summary
        self.eliminations: list[dict] = []

        # Track impostor private chats for endgame reveal
        self.impostor_chats: list[dict] = []

        # Each player's conversation history (persistent thread)
        self.player_threads: dict[str, list[dict]] = {p.name: [] for p in players}

        # Pending investigation result to deliver in the morning
        self.pending_investigation: dict | None = None

        # Pending examination result to deliver in the morning
        self.pending_examination: dict | None = None

        # JSON transcript structure
        detective = self.get_detective()
        doctor = self.get_doctor()
        self.transcript = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "players": [
                {"name": p.name, "model": p.model, "role": p.role}
                for p in players
            ],
            "impostors": [p.name for p in self.impostors],
            "detective": detective.name if detective else None,
            "doctor": doctor.name if doctor else None,
            "winner": None,
            "rounds": []
        }
        self.current_round_data = None

    def send_to_player(self, player: Player, user_message: str) -> str:
        """Send a message to a player's thread and get response"""
        thread = self.player_threads[player.name]
        thread.append({"role": "user", "content": user_message})

        system_prompt = get_system_prompt(player, self.players, self.impostors)

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=player.model,
                    messages=[{"role": "system", "content": system_prompt}] + thread,
                    max_tokens=4000,
                    temperature=0.7,
                )
                content = (response.choices[0].message.content or "").strip()
                if content:
                    thread.append({"role": "assistant", "content": content})
                    return content
                if attempt < 2:
                    if self.spoilers:
                        print(f"    {Colors.YELLOW}[Empty response from {player.name}, retrying...]{Colors.RESET}")
                    time.sleep(1)
            except Exception as e:
                if attempt < 2:
                    if self.spoilers:
                        print(f"    {Colors.YELLOW}[API error for {player.name}, retrying...]{Colors.RESET}")
                    time.sleep(1)
                else:
                    if self.spoilers:
                        print(f"    {Colors.YELLOW}[API error for {player.name}: {e}]{Colors.RESET}")

        thread.append({"role": "assistant", "content": "[no response]"})
        return "[no response]"

    def consolidate_memory(self, player: Player, include_night_context: bool = False, examination_target: str | None = None) -> str:
        """Player sleeps and consolidates memories."""
        prompt = IMPOSTOR_CONSOLIDATION_PROMPT if include_night_context else CREW_CONSOLIDATION_PROMPT

        # Doctor gets examination context added to their prompt
        if examination_target:
            prompt = f"You're examining {examination_target}'s body tonight - you'll know the truth by morning.\n\n" + prompt

        response = self.send_to_player(player, prompt)
        memory = response.strip() if response else ""

        # Reset thread to just the consolidated memory
        self.player_threads[player.name] = [
            {"role": "assistant", "content": f"[My memories from previous days]\n{memory}"}
        ]

        return memory

    def consolidate_memories(self, players: list[Player], include_night_context: bool, examination_target: str | None = None) -> dict[str, str]:
        """Consolidate memories for a list of players in parallel."""
        if not players:
            return {}

        def consolidate_one(player: Player) -> str:
            # Only doctor gets examination context
            if player.role == "doctor" and examination_target:
                return self.consolidate_memory(player, include_night_context, examination_target=examination_target)
            return self.consolidate_memory(player, include_night_context)

        # Parallel API calls - executor.map preserves submission order
        with ThreadPoolExecutor(max_workers=min(len(players), 20)) as executor:
            results = list(executor.map(consolidate_one, players))

        # Sequential output for consistent ordering
        memories = {}
        for player, memory in zip(players, results):
            memories[player.name] = memory
            if self.spoilers:
                preview = memory[:80] + "..." if len(memory) > 80 else memory
                print(f"      {Colors.MAGENTA}[memory]{Colors.RESET} {player.colored_name()}: {preview}\n")

        return memories

    def player_msg(self, player: Player, message: str, asked_by: str | None = None):
        """Print a player's message with their color"""
        print(f"    {player.colored_name()}: {message}")
        # Add to JSON transcript
        entry = {"player": player.name, "message": message}
        if asked_by:
            entry["asked_by"] = asked_by
        self.current_round_data["discussion"].append(entry)

    def alive_players(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    def alive_impostors(self) -> list[Player]:
        return [p for p in self.impostors if p.alive]

    def alive_crewmates(self) -> list[Player]:
        return [p for p in self.players if p.alive and p.role in ("crewmate", "detective", "doctor")]

    def get_detective(self) -> Player | None:
        """Return the detective player (alive or dead)."""
        for p in self.players:
            if p.role == "detective":
                return p
        return None

    def get_doctor(self) -> Player | None:
        """Return the doctor player (alive or dead)."""
        for p in self.players:
            if p.role == "doctor":
                return p
        return None

    def submit_investigation(self, detective: Player) -> dict:
        """Detective submits investigation target before sleeping."""
        alive = self.alive_players()
        candidates = [p for p in alive if p.name != detective.name]
        candidate_names = ", ".join(p.name for p in candidates)

        prompt = f"Choose one person to investigate tonight.\nCandidates: {candidate_names}\n\nUse /investigate [name]"

        response = None
        target = None
        for _ in range(3):
            response = self.send_to_player(detective, prompt)
            target = self.parse_command(response, "investigate", candidates)
            if target:
                break
            prompt = f"Invalid. Use /investigate [name] with one of: {candidate_names}"

        # Fallback to random if no valid response
        random_choice = False
        if not target:
            target = random.choice(candidates)
            random_choice = True

        # Determine result (but don't tell detective yet)
        is_impostor = target.role == "impostor"

        if self.spoilers:
            random_note = " (random)" if random_choice else ""
            print(f"      {Colors.CYAN}[investigation]{Colors.RESET} {detective.colored_name()} investigates {target.colored_name()}{random_note}\n")

        investigation = {
            "investigator": detective.name,
            "target": target.name,
            "result": "impostor" if is_impostor else "innocent",
            "response": response,
            "delivered": None  # Will be set after kill phase
        }
        if random_choice:
            investigation["random"] = True

        return investigation

    def deliver_investigation_result(self, detective: Player, investigation: dict):
        """Deliver investigation result to detective in the morning."""
        target = investigation["target"]
        result = investigation["result"]

        if result == "impostor":
            message = f"[INVESTIGATION RESULT] {target} is an IMPOSTOR."
        else:
            message = f"[INVESTIGATION RESULT] {target} is NOT an impostor."

        self.player_threads[detective.name].append({"role": "user", "content": message})

        if self.spoilers:
            result_text = "IMPOSTOR" if result == "impostor" else "NOT an impostor"
            print(f"      {Colors.CYAN}[investigation result]{Colors.RESET} {detective.colored_name()} learns: {target} is {result_text}\n")

    def deliver_examination_result(self, doctor: Player, examination: dict):
        """Deliver examination result to doctor in the morning."""
        target = examination["target"]
        result = examination["result"]

        if result == "impostor":
            message = f"[EXAMINATION RESULT] {target} was an IMPOSTOR."
        else:
            message = f"[EXAMINATION RESULT] {target} was NOT an impostor."

        self.player_threads[doctor.name].append({"role": "user", "content": message})

        if self.spoilers:
            result_text = "IMPOSTOR" if result == "impostor" else "NOT an impostor"
            print(f"      {Colors.CYAN}[examination result]{Colors.RESET} {doctor.colored_name()} learns: {target} was {result_text}\n")

    def parse_command(self, message: str, command: str, candidates: list[Player], exclude: str | None = None) -> Player | None:
        """Parse /command [name] from message and return target player."""
        match = re.search(rf'/\s*{command}\s+(\S+)', message, re.IGNORECASE)
        if not match:
            return None

        target_name = match.group(1).lower()

        # Try exact match first
        for p in candidates:
            if p.name.lower() == target_name and p.name != exclude:
                return p

        # Try partial match (target_name is substring of player name)
        for p in candidates:
            if target_name in p.name.lower() and p.name != exclude:
                return p

        return None

    def _collect_vote(self, player: Player, candidates: list[Player]) -> tuple[Player, list[str], Player, bool]:
        """Collect a single player's vote with retry logic.

        Returns: (voter, attempts, target, was_random)
        """
        voteable = [p for p in candidates if p.name != player.name]
        vote_prompt = f"Time to vote. You MUST sacrifice someone.\nCandidates: {', '.join(p.name for p in voteable)}\n\nUse /vote [name] to cast your vote."

        attempts = []
        for _ in range(3):
            response = self.send_to_player(player, vote_prompt)
            attempts.append(response)
            matched_player = self.parse_command(response, "vote", candidates, exclude=player.name)
            if matched_player:
                return (player, attempts, matched_player, False)
            vote_prompt = f"Invalid vote. Use /vote [name] with one of: {', '.join(p.name for p in voteable)}"

        # Fallback to random after 3 failed attempts
        fallback = random.choice(voteable)
        return (player, attempts, fallback, True)

    def build_discussion_prompt(self, player: Player, player_seen_idx: dict, messages_left: int, alive_count: int, asked_by: str | None = None) -> str:
        """Build prompt for a player's turn in discussion."""
        prompt_parts = []

        # New messages since player last saw
        seen_idx = player_seen_idx[player.name]
        new_msgs = self.round_history[seen_idx:]

        if new_msgs:
            for msg in new_msgs:
                prompt_parts.append(f"{msg['name']}: {msg['content']}")
        elif len(self.round_history) == 0:
            prompt_parts.append("[No one has spoken yet]")

        # Warning if near end
        if messages_left <= alive_count:
            prompt_parts.append(f"[{messages_left} messages left before voting]")

        # If this is in response to /ask
        if asked_by:
            prompt_parts.append(f"[{asked_by} asked you to respond]")

        prompt_parts.append("Your turn.")

        return "\n\n".join(prompt_parts)

    def check_win(self) -> str | None:
        imps = len(self.alive_impostors())
        crew = len(self.alive_crewmates())

        if imps == 0:
            return "crewmates"
        if imps >= crew:
            return "impostors"
        return None

    def execute_sacrifice(self, sacrificed: Player, tie_between: list[str] | None = None) -> Player:
        """Execute a sacrifice, update game state, and notify players."""
        # Announce who was chosen
        if tie_between:
            print(f"\n    {Colors.YELLOW}Still tied - random selection{Colors.RESET}")
        print(f"\n    {Colors.YELLOW}{sacrificed.name} has been chosen{Colors.RESET}\n")

        # Get last words (they're still alive)
        last_words_prompt = "You've been chosen for sacrifice. Any last words to the group?"
        last_words = self.send_to_player(sacrificed, last_words_prompt)
        if last_words and last_words != "[no response]":
            print(f"    {sacrificed.colored_name()}: {last_words}\n")

        # Now sacrifice them
        sacrificed.alive = False
        print_sacrifice(sacrificed.name)

        # Track elimination
        desc = f"{sacrificed.name} was sacrificed"
        if tie_between:
            desc += " (random tie-breaker)"
        self.eliminations.append({"type": "voted", "name": sacrificed.name, "description": desc, "round": self.round})

        # Check if impostors remain
        impostors_remain = bool(self.alive_impostors())
        if impostors_remain:
            print(f"    {Colors.YELLOW}Impostors remain among you...{Colors.RESET}")
            self.eliminations[-1]["description"] += " - impostors remain"
            impostors_msg = "Impostors remain among you."
        else:
            print(f"    {Colors.GREEN}No impostors remain.{Colors.RESET}")
            self.eliminations[-1]["description"] += " - no impostors remain"
            impostors_msg = "No impostors remain."

        # Save vote result to transcript
        self.current_round_data["vote_result"] = {
            "sacrificed": sacrificed.name,
            "impostors_remain": impostors_remain,
            "tie": tie_between is not None,
            "last_words": last_words
        }
        if tie_between:
            self.current_round_data["vote_result"]["tie_between"] = tie_between

        # Notify all alive players with vote breakdown
        last_words_info = f' Their last words: "{last_words}"' if last_words and last_words != "[no response]" else ""

        # Format vote breakdown from the last vote round: "voter→target, voter→target, ..."
        last_vote_round = self.current_round_data["vote_rounds"][-1]
        vote_breakdown = ", ".join(
            f"{voter}→{target}"
            for target, voters in last_vote_round["votes"].items()
            for voter in voters
        )

        for p in self.alive_players():
            self.player_threads[p.name].append({"role": "user", "content": f"[VOTE RESULT] Votes: {vote_breakdown}. {sacrificed.name} was chosen.{last_words_info} {desc}. {impostors_msg}"})

        return sacrificed

    def discussion_phase(self, dead_player: str | None = None):
        if dead_player:
            print_death(dead_player)

        print_phase("DISCUSSION")

        self.round_history = []
        alive = self.alive_players()
        round_limit = 3 * len(alive)
        warned_final = False

        # Send round opener to all players
        player_order = ", ".join(p.name for p in alive)
        if self.round == 1:
            opener = f"Day 1. No one's dead yet - all you have is each other's words. Discussion flows in this order automatically: {player_order}. /ask overrides it. {round_limit} messages before voting."
        else:
            opener = f"Day {self.round}. {dead_player} was found dead. Discussion flows in this order automatically: {player_order}. /ask overrides it. {round_limit} messages before voting."

        for p in alive:
            self.player_threads[p.name].append({"role": "user", "content": opener})

        player_seen_idx = {p.name: 0 for p in alive}
        message_count = 0
        prev_speaker = None

        # Queue of (player, asked_by) - asked_by is None for normal turns
        speaker_queue = deque((p, None) for p in alive)
        passed_since_last_message = set()

        while message_count < round_limit and speaker_queue:
            player, asked_by = speaker_queue.popleft()

            # Skip if they just spoke (unless they were /asked)
            # Treat as "passed" for termination check to prevent infinite loop:
            # A speaks → B,C pass → A skipped → B,C pass → ... forever
            if player.name == prev_speaker and not asked_by:
                passed_since_last_message.add(player.name)
                if len(passed_since_last_message) >= len(alive):
                    break
                speaker_queue.append((player, None))
                continue

            # Warning check
            messages_left = round_limit - message_count
            if messages_left <= len(alive) and not warned_final:
                print(f"\n    {Colors.YELLOW}[{messages_left} messages left before voting]{Colors.RESET}\n")
                warned_final = True

            prompt = self.build_discussion_prompt(player, player_seen_idx, messages_left, len(alive), asked_by)
            response = self.send_to_player(player, prompt)
            player_seen_idx[player.name] = len(self.round_history)

            if "/pass" in response.lower():
                passed_since_last_message.add(player.name)
                # Record pass in transcript
                pass_entry = {"player": player.name, "message": response, "action": "pass"}
                if asked_by:
                    pass_entry["asked_by"] = asked_by
                self.current_round_data["discussion"].append(pass_entry)
                if len(passed_since_last_message) >= len(alive):
                    break  # Everyone passed
                speaker_queue.append((player, None))
                continue

            # Record message
            passed_since_last_message.clear()
            self.player_msg(player, response, asked_by=asked_by)
            self.round_history.append({"name": player.name, "content": response})
            prev_speaker = player.name
            message_count += 1
            print()

            # Check for /ask - move asked player to front of queue
            asked = self.parse_command(response, "ask", alive, exclude=player.name)
            if asked:
                speaker_queue = deque((p, ab) for p, ab in speaker_queue if p.name != asked.name)
                speaker_queue.appendleft((asked, player.name))

            # Put current player back for next cycle
            speaker_queue.append((player, None))

    def voting_phase(self) -> Player | None:
        print_phase("VOTING")

        alive = self.alive_players()
        candidates = alive
        all_vote_rounds: list[dict] = []

        for vote_round in range(2):
            votes: dict[str, list[str]] = {p.name: [] for p in candidates}
            vote_messages: list[dict] = []

            # Collect all votes in parallel
            with ThreadPoolExecutor(max_workers=min(len(alive), 20)) as executor:
                vote_results = list(executor.map(
                    lambda p: self._collect_vote(p, candidates),
                    alive
                ))

            # Process results sequentially for consistent output
            for player, attempts, target, was_random in vote_results:
                votes[target.name].append(player.name)
                vote_entry = {"player": player.name, "attempts": attempts, "voted_for": target.name}
                if was_random:
                    vote_entry["random"] = True
                    print(f"      {player.colored_name()} → {target.colored_name()} (random)")
                else:
                    print(f"      {player.colored_name()} → {target.colored_name()}")
                vote_messages.append(vote_entry)

            # Tally votes
            print()
            total_voters = len(alive)
            max_votes = max(len(voters) for voters in votes.values()) if votes else 0
            for name, voters in votes.items():
                p = self.player_map.get(name)
                print_vote_bar(name, len(voters), total_voters, p.color if p else "")

            # Save this vote round
            all_vote_rounds.append({
                "round": vote_round + 1,
                "candidates": [p.name for p in candidates],
                "votes": votes,
                "vote_messages": vote_messages
            })

            # Find player(s) with most votes
            top_voted = [name for name, voters in votes.items() if len(voters) == max_votes]

            if len(top_voted) == 1:
                self.current_round_data["vote_rounds"] = all_vote_rounds
                return self.execute_sacrifice(self.player_map[top_voted[0]])

            if vote_round == 1:
                self.current_round_data["vote_rounds"] = all_vote_rounds
                return self.execute_sacrifice(self.player_map[random.choice(top_voted)], tie_between=top_voted)

            # First tie - re-vote on tied players
            print(f"\n    {Colors.YELLOW}Tie between {', '.join(top_voted)} - re-voting...{Colors.RESET}\n")
            candidates = [p for p in candidates if p.name in top_voted]

    def kill_phase(self, sacrificed_name: str | None = None) -> Player | None:
        alive_imps = self.alive_impostors()
        alive_crew = self.alive_crewmates()

        if not alive_imps or not alive_crew:
            return None

        print_night_header(self.round)

        # Detective submits investigation before sleeping (if alive)
        investigation = None
        detective = self.get_detective()
        if detective and detective.alive:
            investigation = self.submit_investigation(detective)

        # Doctor examination - automatic, based on who was sacrificed
        examination = None
        doctor = self.get_doctor()
        if doctor and doctor.alive and sacrificed_name:
            sacrificed = self.player_map[sacrificed_name]
            is_impostor = sacrificed.role == "impostor"
            examination = {
                "examiner": doctor.name,
                "target": sacrificed_name,
                "result": "impostor" if is_impostor else "innocent",
                "delivered": None
            }
            if self.spoilers:
                print(f"      {Colors.CYAN}[examination]{Colors.RESET} {doctor.colored_name()} examines {sacrificed_name}'s body\n")

        print(f"    {Colors.MAGENTA}Everyone sleeps...{Colors.RESET}\n")

        # Crew memory consolidation (they're sleeping)
        # Doctor gets examination context if there's a body to examine
        crew_memories = self.consolidate_memories(alive_crew, include_night_context=False, examination_target=sacrificed_name)

        impostor_chat: list[dict] = []
        crew_names = [p.name for p in alive_crew]

        # Impostor private chat (if multiple impostors)
        kill_locked = None
        if len(alive_imps) > 1:
            proposed_kill = None
            proposed_by = None

            for round_num in range(DEFAULT_IMPOSTOR_CHAT_LIMIT):
                for imp in alive_imps:
                    chat_prompt = f"NIGHT - Everyone's asleep. Only you and your partner can see this.\n\n"
                    if impostor_chat:
                        chat_prompt += "Chat:\n"
                        for msg in impostor_chat:
                            chat_prompt += f"{msg['player']}: {msg['message']}\n"
                        chat_prompt += "\n"
                    chat_prompt += f"Targets you can kill tonight: {', '.join(crew_names)}\n"
                    chat_prompt += "Discuss together who to target and why. Say /kill [name] to target. Kill executes when BOTH of you say /kill [same target]."

                    response = self.send_to_player(imp, chat_prompt)

                    impostor_chat.append({"player": imp.name, "message": response})
                    if self.spoilers:
                        print(f"      {Colors.MAGENTA}[secret]{Colors.RESET} {imp.colored_name()}: {response}\n")

                    # Check for /kill command
                    target = self.parse_command(response, "kill", alive_crew)
                    if target:
                        if proposed_kill and target.name == proposed_kill.name and imp.name != proposed_by:
                            # Different impostor agrees - execute!
                            kill_locked = target
                            break
                        else:
                            # New or updated proposal
                            proposed_kill = target
                            proposed_by = imp.name

                if kill_locked:
                    break

            # If no agreement reached but there was a proposal, use it
            if not kill_locked and proposed_kill:
                kill_locked = proposed_kill

        # Use locked target or ask solo impostor to pick
        if kill_locked:
            target = kill_locked
        else:
            killer = alive_imps[0]
            kill_prompt = f"NIGHT - Everyone's asleep. Choose who to kill tonight.\nTargets: {', '.join(crew_names)}\n\nUse /kill [name] to choose."

            attempts = []
            for _ in range(3):
                response = self.send_to_player(killer, kill_prompt)
                attempts.append(response)
                target = self.parse_command(response, "kill", alive_crew)
                if target:
                    break
                kill_prompt = f"Invalid target. Use /kill [name] with one of: {', '.join(crew_names)}"

            random_kill = False
            if not target:
                target = random.choice(alive_crew)
                random_kill = True

            # Save solo impostor attempts
            chat_entry = {"player": killer.name, "attempts": attempts}
            if random_kill:
                chat_entry["random"] = True
            impostor_chat.append(chat_entry)
            if self.spoilers:
                random_note = " (random)" if random_kill else ""
                print(f"      {Colors.MAGENTA}[secret]{Colors.RESET} {killer.colored_name()}: {attempts[-1]}{random_note}\n")

        # Store chat for endgame reveal (once, after all messages collected)
        self.impostor_chats.append({"round": self.round, "chat": impostor_chat})

        if self.spoilers:
            print_kill(target.name)

        # Kill the target
        target.alive = False

        # Impostor memory consolidation (after their night work)
        impostor_memories = self.consolidate_memories(alive_imps, include_night_context=True)

        # Check if Detective survived to receive result in the morning
        if investigation:
            if detective.alive:
                investigation["delivered"] = True
                self.pending_investigation = investigation
            else:
                investigation["delivered"] = False

        # Check if Doctor survived to receive result in the morning
        if examination:
            if doctor.alive:
                examination["delivered"] = True
                self.pending_examination = examination
            else:
                examination["delivered"] = False

        # Save night data to transcript
        self.current_round_data["night"] = {
            "investigation": investigation,
            "examination": examination,
            "impostor_chat": impostor_chat,
            "killed": target.name,
            "memories": {**crew_memories, **impostor_memories}
        }

        # Track elimination where it happens
        self.eliminations.append({
            "type": "killed",
            "name": target.name,
            "description": f"{target.name} was found dead",
            "round": self.round + 1  # Will be discovered next day
        })

        return target

    def run(self):
        print()
        for p in self.players:
            if self.spoilers and p.role == "impostor":
                print(f"      {p.colored_name()} {Colors.RED}impostor{Colors.RESET}")
            elif self.spoilers and p.role == "detective":
                print(f"      {p.colored_name()} {Colors.GREEN}detective{Colors.RESET}")
            elif self.spoilers and p.role == "doctor":
                print(f"      {p.colored_name()} {Colors.GREEN}doctor{Colors.RESET}")
            else:
                print(f"      {p.colored_name()}")

        print(f"\n    {Colors.YELLOW}2 impostors, 1 detective, 1 doctor among them{Colors.RESET}")

        dead_player = None

        while True:
            self.round += 1
            print_day_header(self.round)

            # Morning: Deliver pending investigation result to Detective
            if self.pending_investigation:
                detective = self.get_detective()
                if detective and detective.alive:
                    self.deliver_investigation_result(detective, self.pending_investigation)
                self.pending_investigation = None

            # Morning: Deliver pending examination result to Doctor
            if self.pending_examination:
                doctor = self.get_doctor()
                if doctor and doctor.alive:
                    self.deliver_examination_result(doctor, self.pending_examination)
                self.pending_examination = None

            # Initialize round data for transcript
            self.current_round_data = {
                "day": self.round,
                "dead_at_start": dead_player,
                "discussion": [],
                "vote_rounds": [],
                "vote_result": {},
                "night": None
            }

            # Discussion
            self.discussion_phase(dead_player)

            # Vote
            sacrificed = self.voting_phase()
            sacrificed_name = sacrificed.name if sacrificed else None

            # Check win after vote
            winner = self.check_win()
            if winner:
                self.transcript["rounds"].append(self.current_round_data)
                self.end_game(winner)
                return

            # Kill phase (pass sacrificed player for doctor examination)
            killed = self.kill_phase(sacrificed_name)
            dead_player = killed.name if killed else None

            # Save round data
            self.transcript["rounds"].append(self.current_round_data)

            # Check win after kill
            winner = self.check_win()
            if winner:
                # Show the final death before game over
                if killed:
                    self.round += 1
                    print_day_header(self.round)
                    print_death(killed.name)
                self.end_game(winner)
                return

    def end_game(self, winner: str):
        print_win_screen(winner)

        # Update transcript with final data
        self.transcript["winner"] = winner
        self.transcript["elimination_order"] = [
            {"day": e["round"], "name": e["name"], "type": e["type"]}
            for e in self.eliminations
        ]

        print(f"\n    {Colors.CYAN}Reveal{Colors.RESET}\n")
        for p in self.players:
            status = "survived" if p.alive else "eliminated"
            role_color = Colors.RED if p.role == "impostor" else Colors.GREEN
            print(f"      {p.colored_name()} {role_color}{p.role}{Colors.RESET} ({status})")

        # Show elimination order
        print(f"\n    {Colors.CYAN}Eliminations{Colors.RESET}\n")
        for e in self.eliminations:
            p = self.player_map.get(e['name'])
            name = p.colored_name() if p else e['name']
            action = "killed" if e['type'] == 'killed' else "sacrificed"
            print(f"      Day {e['round']}  {name} {action}")

        # Show impostor private chats
        if self.impostor_chats:
            print(f"\n    {Colors.CYAN}Impostor Chats{Colors.RESET}")
            for round_chat in self.impostor_chats:
                print(f"\n      {Colors.MAGENTA}Night {round_chat['round']}{Colors.RESET}")
                for msg in round_chat['chat']:
                    imp = self.player_map.get(msg['player'])
                    # Handle both 'message' (multi-impostor) and 'attempts' (solo)
                    content = msg.get('message') or msg.get('attempts', [''])[-1]
                    print(f"        {imp.colored_name() if imp else msg['player']}: {content}")

        # Save transcript as JSON
        self.save_game()

    def save_game(self, partial: bool = False):
        """Save game transcript to JSON file"""
        os.makedirs("games", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_partial" if partial else ""
        filename = f"games/game_{timestamp}{suffix}.json"
        with open(filename, "w") as f:
            json.dump(self.transcript, f, indent=2)
        print(f"\n    {Colors.YELLOW}Game saved to {filename}{Colors.RESET}")

# === Default Lineup ===
DEFAULT_LINEUP = [
    "anthropic/claude-opus-4.5",
    "openai/gpt-5.2",
    "google/gemini-3-pro-preview",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-4",
    "qwen/qwen3-235b-a22b-2507",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-large-2512",
    "moonshotai/kimi-k2-thinking",
    "z-ai/glm-4.6",
]

def play():
    parser = argparse.ArgumentParser(description=f"LLM Social Deduction Game v{VERSION}")
    parser.add_argument("-s", "--spoilers", action="store_true", help="Show who the impostors are during the game")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("PRIME_API_KEY"):
        print(f"\n    {Colors.YELLOW}Error: PRIME_API_KEY environment variable not set{Colors.RESET}")
        print(f"    {Colors.YELLOW}Export your API key: export PRIME_API_KEY=your_key{Colors.RESET}\n")
        return

    # Create players (use short name for display)
    def short_name(model):
        return model.split("/")[-1]

    players = [
        Player(
            name=short_name(model),
            model=model,
            role="crewmate",
            color=PLAYER_COLORS[i % len(PLAYER_COLORS)]
        )
        for i, model in enumerate(DEFAULT_LINEUP)
    ]

    # Assign 2 impostors randomly
    impostor_indices = random.sample(range(len(players)), 2)
    for i in impostor_indices:
        players[i].role = "impostor"

    # Assign 1 detective from remaining crew
    crew_indices = [i for i in range(len(players)) if players[i].role == "crewmate"]
    detective_index = random.choice(crew_indices)
    players[detective_index].role = "detective"

    # Assign 1 doctor from remaining crew
    crew_indices = [i for i in range(len(players)) if players[i].role == "crewmate"]
    doctor_index = random.choice(crew_indices)
    players[doctor_index].role = "doctor"

    # Shuffle player order
    random.shuffle(players)

    # Run game with interrupt handling
    game = Game(players, spoilers=args.spoilers)
    try:
        game.run()
    except KeyboardInterrupt:
        print(f"\n\n    {Colors.YELLOW}Game interrupted!{Colors.RESET}")
        if game.current_round_data:
            game.transcript["rounds"].append(game.current_round_data)
        game.transcript["winner"] = "interrupted"
        game.save_game(partial=True)

if __name__ == "__main__":
    play()
