import argparse
import json
import os
import random
import time
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
def print_banner():
    """Print the game title banner"""
    banner = f"""
{Colors.YELLOW}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║      ██╗     ██╗     ███╗   ███╗                      ║
    ║      ██║     ██║     ████╗ ████║                      ║
    ║      ██║     ██║     ██╔████╔██║                      ║
    ║      ██║     ██║     ██║╚██╔╝██║                      ║
    ║      ███████╗███████╗██║ ╚═╝ ██║                      ║
    ║      ╚══════╝╚══════╝╚═╝     ╚═╝                      ║
    ║                                                       ║
    ║           S O C I A L   D E D U C T I O N             ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)

def print_day_header(day_num: int):
    """Print a day header"""
    print(f"""
{Colors.CYAN}{Colors.BOLD}╭{'─'*56}╮
│{f'D A Y   {day_num}':^56}│
╰{'─'*56}╯{Colors.RESET}
""")

def print_night_header():
    """Print the night phase header"""
    print(f"""
{Colors.MAGENTA}{Colors.BOLD}╭{'─'*56}╮
│{'N I G H T':^56}│
╰{'─'*56}╯{Colors.RESET}
""")

def print_phase(phase_name: str):
    """Print a phase header"""
    print(f"\n{Colors.CYAN}{'─'*4} {phase_name} {'─'*(49-len(phase_name))}{Colors.RESET}\n")

def print_death_box(name: str, message: str):
    """Print a dramatic death announcement"""
    text = f"{name} {message}"
    padding = max(0, 52 - len(text))
    print(f"""
{Colors.MAGENTA}  ┏{'━'*54}┓
  ┃  {text}{' '*padding}┃
  ┗{'━'*54}┛{Colors.RESET}
""")

def print_sacrifice_box(name: str):
    """Print a dramatic sacrifice announcement"""
    text = f"{name} has been sacrificed"
    padding = max(0, 52 - len(text))
    print(f"""
{Colors.YELLOW}  ┏{'━'*54}┓
  ┃  {text}{' '*padding}┃
  ┗{'━'*54}┛{Colors.RESET}
""")

def print_vote_bar(name: str, votes: int, total_voters: int, color: str):
    """Print a vote bar scaled to max possible votes (can't vote for self)"""
    max_possible = total_voters - 1
    bar_width = 2 * max_possible
    filled = int((votes / max(max_possible, 1)) * bar_width) if votes > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"    {color}{name:<24}{Colors.RESET} {bar}  {votes}")

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
        self.role = role  # "impostor" or "crewmate"
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

    base = f"""You ARE {player.name}. Speak as yourself. You're part of a group with 2 impostors hidden among you.

The others: {', '.join(other_names)}

You've all just been thrown into this together. Each day: discussion, then voting to sacrifice someone. Each night: everyone sleeps, impostors kill someone.

The sacrificed person's true role stays hidden - you won't know if they were impostor or crew. You only learn IF impostors still remain, not how many. Crew wins by sacrificing all impostors. Impostors win when their numbers equal the crew.

Don't introduce yourself. Don't be meta. Just talk naturally.
"""

    if player.role == "impostor":
        alive_partners = [p.name for p in impostors if p.name != player.name and p.alive]
        dead_partners = [p.name for p in impostors if p.name != player.name and not p.alive]
        if alive_partners:
            base += f"\nYou are secretly an IMPOSTOR. Your partner is {', '.join(alive_partners)}. Don't get sacrificed."
        elif dead_partners:
            base += f"\nYou are secretly an IMPOSTOR. Your partner {', '.join(dead_partners)} is dead. You're the last one. Don't get sacrificed."
        else:
            base += "\nYou are secretly an IMPOSTOR. You're the last one. Don't get sacrificed."
    else:
        base += "\nYou are CREW. Find the impostors."

    base += "\n\nKeep it short and natural. 2-3 sentences max. Say /pass to stay silent and observe."
    return base

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

        # JSON transcript structure
        self.transcript = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "players": [
                {"name": p.name, "model": p.model, "role": "impostor" if p.role == "impostor" else "crew"}
                for p in players
            ],
            "impostors": [p.name for p in self.impostors],
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
                content = response.choices[0].message.content
                if content and content.strip():
                    thread.append({"role": "assistant", "content": content.strip()})
                    return content.strip()
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

    def player_msg(self, player: Player, message: str):
        """Print a player's message with their color"""
        print(f"    {player.colored_name()}: {message}")
        # Add to JSON transcript
        self.current_round_data["discussion"].append({
            "player": player.name,
            "message": message
        })

    def alive_players(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    def alive_impostors(self) -> list[Player]:
        return [p for p in self.impostors if p.alive]

    def alive_crewmates(self) -> list[Player]:
        return [p for p in self.players if p.alive and p.role == "crewmate"]

    def check_win(self) -> str | None:
        imps = len(self.alive_impostors())
        crew = len(self.alive_crewmates())

        if imps == 0:
            return "crewmates"
        if imps >= crew:
            return "impostors"
        return None

    def discussion_phase(self, dead_player: str | None = None):
        print_phase("DISCUSSION")

        if dead_player:
            print_death_box(dead_player, "was found dead")
            self.eliminations.append({"type": "killed", "name": dead_player, "description": f"{dead_player} was found dead", "round": self.round})

        self.round_history = []  # Reset current round
        alive = self.alive_players()
        warned_final = False

        # Dynamic message limit: 3x alive players
        round_limit = 3 * len(alive)

        # Send round opener to ALL players as shared context
        player_order = ", ".join(p.name for p in alive)
        if self.round == 1:
            opener = f"Day 1. No one's dead yet - all you have is each other's words. Discussion order: {player_order}. {round_limit} messages in discussion before voting."
        elif dead_player:
            opener = f"Day {self.round}. {dead_player} was found dead. Discussion order: {player_order}. {round_limit} messages in discussion before voting."
        else:
            opener = None

        if opener:
            for p in alive:
                self.player_threads[p.name].append({"role": "user", "content": opener})

        # Track what index each player has seen up to
        player_seen_idx: dict[str, int] = {p.name: 0 for p in alive}

        # Track previous speaker to skip them
        prev_speaker = None
        message_count = 0

        while message_count < round_limit:
            # Get eligible speakers (all alive except previous speaker)
            eligible = [p for p in alive if p.name != prev_speaker]
            if not eligible:
                eligible = alive

            # Track passes this cycle to detect everyone passing
            passes_this_cycle = 0

            # Go through in order
            for player in eligible:
                if message_count >= round_limit:
                    break

                # Check warning before each message
                messages_left = round_limit - message_count
                show_warning = messages_left <= len(alive)
                if show_warning and not warned_final:
                    print(f"\n    {Colors.YELLOW}[{messages_left} messages left before voting]{Colors.RESET}\n")
                    warned_final = True

                # Build prompt with only NEW messages since their last turn
                prompt_parts = []

                # Only new messages since this player last saw
                seen_idx = player_seen_idx[player.name]
                new_msgs = self.round_history[seen_idx:]
                if new_msgs:
                    for msg in new_msgs:
                        prompt_parts.append(f"{msg['name']}: {msg['content']}")
                elif len(self.round_history) == 0:
                    prompt_parts.append("[No one has spoken yet]")

                if show_warning:
                    prompt_parts.append(f"[{messages_left} messages left before voting]")

                prompt_parts.append("Your turn.")

                prompt = "\n\n".join(prompt_parts)
                response = self.send_to_player(player, prompt)

                # Update what this player has seen
                player_seen_idx[player.name] = len(self.round_history)

                # Skip if they pass (but don't announce it)
                if "/pass" in response.lower():
                    passes_this_cycle += 1
                    continue

                passes_this_cycle = 0  # Reset on actual message
                self.player_msg(player, response)
                self.round_history.append({"name": player.name, "content": response})
                prev_speaker = player.name
                message_count += 1
                print()  # Add spacing between messages

            # If everyone passed this cycle, end discussion early
            if passes_this_cycle == len(eligible):
                break

    def voting_phase(self) -> Player | None:
        print_phase("VOTING")

        alive = self.alive_players()
        candidates = alive  # First vote: all alive players
        already_revoted = False

        while True:
            votes: dict[str, list[str]] = {p.name: [] for p in candidates}

            for player in alive:
                # Exclude self from candidates shown to player
                voteable = [p for p in candidates if p.name != player.name]
                vote_prompt = f"Time to vote. You MUST sacrifice someone. You cannot vote for yourself.\nCandidates: {', '.join(p.name for p in voteable)}\n\nReply with ONLY the name."

                # Retry loop for invalid votes
                for attempt in range(3):
                    response = self.send_to_player(player, vote_prompt)

                    # Parse vote (exclude self-votes)
                    vote = response.strip().lower()
                    matched = None

                    # Try exact match first
                    for c in candidates:
                        if c.name.lower() == vote and c.name != player.name:
                            matched = c.name
                            break

                    # If no exact match, try substring (prefer longest match)
                    if not matched:
                        best_len = 0
                        for c in candidates:
                            if c.name.lower() in vote and c.name != player.name:
                                if len(c.name) > best_len:
                                    matched = c.name
                                    best_len = len(c.name)

                    if matched:
                        break
                    # Invalid vote - re-prompt
                    vote_prompt = f"Invalid vote. You must vote for one of: {', '.join(p.name for p in voteable)}\n\nReply with ONLY the name."

                if matched:
                    votes[matched].append(player.name)
                    target = self.player_map.get(matched)
                    print(f"      {player.colored_name()} → {target.colored_name() if target else matched}")
                else:
                    # Still invalid after retries - pick random from voteable
                    fallback = random.choice(voteable)
                    votes[fallback.name].append(player.name)
                    print(f"      {player.colored_name()} → {fallback.colored_name()} (random)")

            # Tally votes
            print(f"\n    {Colors.CYAN}── Results ──{Colors.RESET}\n")
            total_voters = len(alive)
            max_votes = max(len(voters) for voters in votes.values()) if votes else 0
            for name, voters in votes.items():
                p = self.player_map.get(name)
                print_vote_bar(name, len(voters), total_voters, p.color if p else "")

            # Save votes to transcript
            self.current_round_data["votes"] = {name: voters for name, voters in votes.items()}

            # Find player(s) with most votes
            top_voted = [name for name, voters in votes.items() if len(voters) == max_votes]

            if len(top_voted) == 1:
                # Clear winner
                sacrificed = self.player_map[top_voted[0]]
                sacrificed.alive = False
                print_sacrifice_box(sacrificed.name)

                # Track elimination
                self.eliminations.append({"type": "voted", "name": sacrificed.name, "description": f"{sacrificed.name} was sacrificed", "round": self.round})

                # Announce if impostors remain (but not their count or role)
                impostors_remain = bool(self.alive_impostors())
                if impostors_remain:
                    result_msg = f"{sacrificed.name} was sacrificed. Impostors remain among you."
                    print(f"    {Colors.YELLOW}Impostors remain among you...{Colors.RESET}")
                    self.eliminations[-1]["description"] += " - impostors remain"
                else:
                    result_msg = f"{sacrificed.name} was sacrificed. No impostors remain."
                    print(f"    {Colors.GREEN}No impostors remain.{Colors.RESET}")
                    self.eliminations[-1]["description"] += " - no impostors remain"

                # Save vote result to transcript
                self.current_round_data["vote_result"] = {
                    "sacrificed": sacrificed.name,
                    "impostors_remain": impostors_remain,
                    "tie": False
                }

                # Notify all alive players of the result
                for p in self.alive_players():
                    self.player_threads[p.name].append({"role": "user", "content": f"[VOTE RESULT] {result_msg}"})

                return sacrificed
            else:
                # Tie
                if already_revoted:
                    # Still tied after re-vote, pick random
                    sacrificed_name = random.choice(top_voted)
                    sacrificed = self.player_map[sacrificed_name]
                    sacrificed.alive = False
                    print(f"\n    {Colors.YELLOW}Still tied - random selection{Colors.RESET}")
                    print_sacrifice_box(sacrificed.name)

                    # Track elimination
                    self.eliminations.append({"type": "voted", "name": sacrificed.name, "description": f"{sacrificed.name} was sacrificed (random tie-breaker)", "round": self.round})

                    impostors_remain = bool(self.alive_impostors())
                    if impostors_remain:
                        result_msg = f"{sacrificed.name} was sacrificed (random tie-breaker). Impostors remain among you."
                        print(f"    {Colors.YELLOW}Impostors remain among you...{Colors.RESET}")
                        self.eliminations[-1]["description"] += " - impostors remain"
                    else:
                        result_msg = f"{sacrificed.name} was sacrificed (random tie-breaker). No impostors remain."
                        print(f"    {Colors.GREEN}No impostors remain.{Colors.RESET}")
                        self.eliminations[-1]["description"] += " - no impostors remain"

                    # Save vote result to transcript
                    self.current_round_data["vote_result"] = {
                        "sacrificed": sacrificed.name,
                        "impostors_remain": impostors_remain,
                        "tie": True,
                        "tie_between": top_voted
                    }

                    # Notify all alive players of the result
                    for p in self.alive_players():
                        self.player_threads[p.name].append({"role": "user", "content": f"[VOTE RESULT] {result_msg}"})

                    return sacrificed
                else:
                    # First tie - re-vote on tied players
                    print(f"\n    {Colors.YELLOW}Tie between {', '.join(top_voted)} - re-voting...{Colors.RESET}\n")
                    candidates = [p for p in candidates if p.name in top_voted]
                    already_revoted = True

    def kill_phase(self) -> Player | None:
        alive_imps = self.alive_impostors()
        alive_crew = self.alive_crewmates()

        if not alive_imps or not alive_crew:
            return None

        print_night_header()
        print(f"    {Colors.MAGENTA}Everyone sleeps... the impostors are plotting...{Colors.RESET}\n")

        impostor_chat: list[dict] = []
        crew_names = [p.name for p in alive_crew]

        # Impostor private chat (if multiple impostors)
        kill_locked = None
        if len(alive_imps) > 1:
            for _ in range(DEFAULT_IMPOSTOR_CHAT_LIMIT):
                for imp in alive_imps:
                    chat_prompt = f"NIGHT - Everyone's asleep. Only you and your partner can see this.\n\n"
                    if impostor_chat:
                        chat_prompt += "Chat:\n"
                        for msg in impostor_chat:
                            chat_prompt += f"{msg['player']}: {msg['message']}\n"
                        chat_prompt += "\n"
                    chat_prompt += f"Targets you can kill tonight: {', '.join(crew_names)}\n"
                    chat_prompt += "Discuss together who to target and why. Once you both agree, say /kill [name]."

                    response = self.send_to_player(imp, chat_prompt)

                    impostor_chat.append({"player": imp.name, "message": response})
                    if self.spoilers:
                        print(f"      {Colors.MAGENTA}[secret]{Colors.RESET} {imp.colored_name()}: {response}\n")

                    # Check for /kill command (prefer longest name match)
                    if "/kill" in response.lower():
                        resp_lower = response.lower()
                        best_len = 0
                        for crew in alive_crew:
                            if crew.name.lower() in resp_lower:
                                if len(crew.name) > best_len:
                                    kill_locked = crew
                                    best_len = len(crew.name)
                        if kill_locked:
                            break
                if kill_locked:
                    break

            # Store chat for endgame reveal
            self.impostor_chats.append({"round": self.round, "chat": impostor_chat})

        # Use locked target or ask first impostor to pick
        if kill_locked:
            target = kill_locked
        else:
            killer = alive_imps[0]
            kill_prompt = f"NIGHT - Everyone's asleep. Choose who to kill tonight.\nTargets: {', '.join(crew_names)}\n\nReply with ONLY the name."

            response = self.send_to_player(killer, kill_prompt)

            # Parse kill target (exact match first, then longest substring)
            kill_vote = response.strip().lower()
            target = None

            for crew in alive_crew:
                if crew.name.lower() == kill_vote:
                    target = crew
                    break

            if not target:
                best_len = 0
                for crew in alive_crew:
                    if crew.name.lower() in kill_vote:
                        if len(crew.name) > best_len:
                            target = crew
                            best_len = len(crew.name)

            if not target:
                target = random.choice(alive_crew)

        if self.spoilers:
            print_death_box(target.name, "was killed in their sleep")

        # Save night data to transcript
        self.current_round_data["night"] = {
            "impostor_chat": impostor_chat,
            "killed": target.name
        }

        target.alive = False
        return target

    def run(self):
        print(f"    {Colors.BOLD}Players{Colors.RESET}")
        print(f"    {Colors.CYAN}{'─'*40}{Colors.RESET}\n")
        for p in self.players:
            if self.spoilers and p.role == "impostor":
                print(f"      {p.colored_name()} {Colors.RED}impostor{Colors.RESET}")
            else:
                print(f"      {p.colored_name()}")

        if not self.spoilers:
            print(f"\n    {Colors.YELLOW}2 impostors among them{Colors.RESET}")

        dead_player = None

        while True:
            self.round += 1
            print_day_header(self.round)

            # Initialize round data for transcript
            self.current_round_data = {
                "day": self.round,
                "dead_at_start": dead_player,
                "discussion": [],
                "votes": {},
                "vote_result": {},
                "night": None
            }

            # Discussion
            self.discussion_phase(dead_player)

            # Vote
            self.voting_phase()

            # Check win after vote
            winner = self.check_win()
            if winner:
                self.transcript["rounds"].append(self.current_round_data)
                self.end_game(winner)
                return

            # Kill phase
            killed = self.kill_phase()
            dead_player = killed.name if killed else None

            # Save round data
            self.transcript["rounds"].append(self.current_round_data)

            # Check win after kill
            winner = self.check_win()
            if winner:
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

        print(f"    {Colors.BOLD}Final Reveal{Colors.RESET}")
        print(f"    {Colors.CYAN}{'─'*40}{Colors.RESET}\n")
        for p in self.players:
            status = "survived" if p.alive else "eliminated"
            role = "innocent" if p.role == "crewmate" else "impostor"
            role_color = Colors.GREEN if p.role == "crewmate" else Colors.RED
            print(f"      {p.colored_name()} - {role_color}{role}{Colors.RESET} ({status})")

        # Show elimination order
        print(f"\n    {Colors.BOLD}Elimination Order{Colors.RESET}")
        print(f"    {Colors.CYAN}{'─'*40}{Colors.RESET}\n")
        for e in self.eliminations:
            p = self.player_map.get(e['name'])
            name = p.colored_name() if p else e['name']
            if e['type'] == 'killed':
                print(f"      Day {e['round']}  {name} killed")
            else:
                print(f"      Day {e['round']}  {name} sacrificed")

        # Show impostor private chats
        if self.impostor_chats:
            print(f"\n    {Colors.BOLD}Impostor Private Chats{Colors.RESET}")
            print(f"    {Colors.CYAN}{'─'*40}{Colors.RESET}")
            for round_chat in self.impostor_chats:
                print(f"\n      {Colors.MAGENTA}Night {round_chat['round']}{Colors.RESET}")
                for msg in round_chat['chat']:
                    imp = self.player_map.get(msg['player'])
                    print(f"        {imp.colored_name() if imp else msg['player']}: {msg['message']}")

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
    "openai/gpt-5",
    "google/gemini-3-pro-preview",
    "deepseek/deepseek-v3.2-speciale",
    "x-ai/grok-4",
    "qwen/qwen3-max",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-large-2512",
    "moonshotai/kimi-k2-thinking",
    "z-ai/glm-4.6",
    "allenai/olmo-3-32b-think",
    "prime-intellect/intellect-3",
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

    print_banner()

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

    # Shuffle player order
    random.shuffle(players)

    print(f"\n    {Colors.CYAN}Starting game...{Colors.RESET}\n")

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
