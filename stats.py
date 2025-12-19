#!/usr/bin/env python3
"""
LLM Social Deduction - ELO & Stats Calculator

Calculates role-based ELO ratings and comprehensive statistics
for LLM players from game transcripts.

ELO System:
- Team vs Team: Your team's average ELO vs opponent team's average ELO
- Everyone on the same team gets the same rating change
- Two separate ratings: impostor_elo and crew_elo
- Overall ELO is weighted average by games played in each role
"""

import json
import os
import glob
import re
from datetime import datetime

# ELO Parameters
STARTING_ELO = 1000
BASE_K_FACTOR = 32


def safe_div(num: float, denom: float, default=None) -> float | None:
    """Safely divide, returning default if denominator is 0."""
    return num / denom if denom else default


def expected_score(team_elo: float, opponent_elo: float) -> float:
    """Calculate expected score (probability of winning) using ELO formula."""
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def calculate_elo_change(team_elo: float, opponent_elo: float, won: bool, k: float) -> float:
    """Calculate ELO rating change based on team avg vs opponent team avg."""
    expected = expected_score(team_elo, opponent_elo)
    actual = 1.0 if won else 0.0
    return k * (actual - expected)


def load_games(games_dir: str = "games") -> list[dict]:
    """Load all completed game JSON files, sorted by timestamp."""
    game_files = glob.glob(os.path.join(games_dir, "game_*.json"))

    # Filter out partial games
    game_files = [f for f in game_files if "_partial" not in f]

    games = []
    for filepath in game_files:
        with open(filepath, "r") as f:
            game = json.load(f)
            game["_filepath"] = filepath
            games.append(game)

    # Sort by timestamp
    games.sort(key=lambda g: g.get("timestamp", ""))

    return games


def initialize_player_stats() -> dict:
    """Create a fresh player stats dictionary."""
    return {
        # ELO ratings
        "impostor_elo": STARTING_ELO,
        "crew_elo": STARTING_ELO,

        # Game counts
        "games_played": 0,
        "wins": 0,
        "losses": 0,

        # Role breakdown
        "impostor_games": 0,
        "impostor_wins": 0,
        "crew_games": 0,
        "crew_wins": 0,

        # Survival stats
        "games_survived": 0,
        "impostor_survived": 0,
        "crew_survived": 0,
        "total_days_survived": 0,
        "times_voted_out": 0,
        "times_killed": 0,

        # Voting stats
        "votes_cast_as_crew": 0,
        "votes_for_impostors": 0,
        "times_received_votes": 0,

        # Activity
        "total_messages": 0,
    }


def get_player_role(game: dict, player_name: str) -> str | None:
    """Get the role of a player in a game."""
    for player in game["players"]:
        if player["name"] == player_name:
            return player["role"]
    return None


def get_elimination_info(game: dict, player_name: str) -> tuple[int, str] | None:
    """Get elimination day and type for a player, or None if survived."""
    for elim in game.get("elimination_order", []):
        if elim["name"] == player_name:
            return elim["day"], elim["type"]
    return None


def count_days_in_game(game: dict) -> int:
    """Count total days in a game."""
    return len(game.get("rounds", []))


def process_game(game: dict, players: dict[str, dict]) -> None:
    """Process a single game and update player stats and ELO."""
    winner = game["winner"]
    impostors = set(game["impostors"])
    all_players = [p["name"] for p in game["players"]]
    crew_members = [p for p in all_players if p not in impostors]

    total_days = count_days_in_game(game)

    # Ensure all players exist in our stats
    for player in all_players:
        if player not in players:
            players[player] = initialize_player_stats()

    # Calculate ELO changes
    # Team vs Team: team avg elo vs team avg elo
    impostor_won = (winner == "impostors")

    # Get current ELOs for calculations
    impostor_elos = {p: players[p]["impostor_elo"] for p in impostors}
    crew_elos = {p: players[p]["crew_elo"] for p in crew_members}

    # Calculate team averages
    avg_impostor_elo = sum(impostor_elos.values()) / len(impostor_elos) if impostor_elos else STARTING_ELO
    avg_crew_elo = sum(crew_elos.values()) / len(crew_elos) if crew_elos else STARTING_ELO

    # Calculate ELO changes based on team avg vs team avg
    # Same change for everyone on the same team
    impostor_change = calculate_elo_change(
        avg_impostor_elo,
        avg_crew_elo,
        impostor_won,
        BASE_K_FACTOR
    )
    crew_change = calculate_elo_change(
        avg_crew_elo,
        avg_impostor_elo,
        not impostor_won,  # Crew wins when impostors lose
        BASE_K_FACTOR
    )

    # Apply same change to all team members
    for imp in impostors:
        players[imp]["impostor_elo"] += impostor_change

    for crew in crew_members:
        players[crew]["crew_elo"] += crew_change

    # Update stats for all players
    for player_name in all_players:
        stats = players[player_name]
        role = get_player_role(game, player_name)
        player_won = (role == "impostor" and impostor_won) or (role == "crew" and not impostor_won)

        # Basic stats
        stats["games_played"] += 1
        if player_won:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        # Role breakdown
        if role == "impostor":
            stats["impostor_games"] += 1
            if player_won:
                stats["impostor_wins"] += 1
        else:
            stats["crew_games"] += 1
            if player_won:
                stats["crew_wins"] += 1

        # Survival stats
        elim_info = get_elimination_info(game, player_name)
        if elim_info is None:
            stats["games_survived"] += 1
            stats["total_days_survived"] += total_days
            if role == "impostor":
                stats["impostor_survived"] += 1
            else:
                stats["crew_survived"] += 1
        else:
            elim_day, elim_type = elim_info
            stats["total_days_survived"] += elim_day
            if elim_type == "voted":
                stats["times_voted_out"] += 1
            elif elim_type == "killed":
                stats["times_killed"] += 1

    # Process voting and message stats from rounds
    for round_data in game.get("rounds", []):
        # Count messages
        for msg in round_data.get("discussion", []):
            speaker = msg["player"]
            if speaker in players:
                players[speaker]["total_messages"] += 1

        # Process votes
        votes = round_data.get("votes", {})
        for target, voters in votes.items():
            # Count votes received
            if target in players:
                players[target]["times_received_votes"] += len(voters)

            # Count voting accuracy (for crew members voting)
            for voter in voters:
                if voter in players:
                    voter_role = get_player_role(game, voter)
                    if voter_role == "crew":
                        players[voter]["votes_cast_as_crew"] += 1
                        if target in impostors:
                            players[voter]["votes_for_impostors"] += 1


def calculate_derived_stats(players: dict[str, dict]) -> dict[str, dict]:
    """Calculate derived statistics and format for output."""
    results = {}

    for name, stats in players.items():
        games = stats["games_played"]
        imp_games = stats["impostor_games"]
        crew_games = stats["crew_games"]
        days_spoken = stats["total_days_survived"] - stats["times_killed"]

        result = {
            # ELO ratings
            "impostor_elo": round(stats["impostor_elo"]),
            "crew_elo": round(stats["crew_elo"]),
            "overall_elo": None,

            # Game counts
            "games_played": games,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": safe_div(stats["wins"], games, 0),

            # Role breakdown
            "impostor_games": imp_games,
            "impostor_wins": stats["impostor_wins"],
            "impostor_win_rate": safe_div(stats["impostor_wins"], imp_games),
            "crew_games": crew_games,
            "crew_wins": stats["crew_wins"],
            "crew_win_rate": safe_div(stats["crew_wins"], crew_games),

            # Survival stats
            "survival_rate": safe_div(stats["games_survived"], games, 0),
            "impostor_survival_rate": safe_div(stats["impostor_survived"], imp_games),
            "crew_survival_rate": safe_div(stats["crew_survived"], crew_games),
            "avg_days_survived": safe_div(stats["total_days_survived"], games, 0),
            "times_voted_out": stats["times_voted_out"],
            "times_killed": stats["times_killed"],

            # Voting accuracy
            "votes_cast_as_crew": stats["votes_cast_as_crew"],
            "votes_for_impostors": stats["votes_for_impostors"],
            "voting_accuracy": safe_div(stats["votes_for_impostors"], stats["votes_cast_as_crew"]),

            # Vote reception
            "times_received_votes": stats["times_received_votes"],
            "avg_votes_against_per_game": safe_div(stats["times_received_votes"], games, 0),

            # Activity
            "total_messages": stats["total_messages"],
            "avg_messages_per_day": safe_div(stats["total_messages"], days_spoken, 0),
        }

        # Calculate overall ELO (weighted average by games in each role)
        if stats["impostor_games"] > 0 and stats["crew_games"] > 0:
            result["overall_elo"] = round(
                (stats["impostor_elo"] * stats["impostor_games"] +
                 stats["crew_elo"] * stats["crew_games"]) / stats["games_played"]
            )
        elif stats["impostor_games"] > 0:
            result["overall_elo"] = round(stats["impostor_elo"])
        elif stats["crew_games"] > 0:
            result["overall_elo"] = round(stats["crew_elo"])

        results[name] = result

    return results


class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def visual_len(s: str) -> int:
    """Get visual length of string, excluding ANSI escape codes."""
    return len(re.sub(r'\033\[[0-9;]*m', '', s))


def pad_right(s: str, width: int) -> str:
    """Pad string to width based on visual length."""
    return s + ' ' * (width - visual_len(s))


def format_with_roles(total: str, imp: str, crew: str, total_width: int, col_width: int) -> str:
    """Format as 'total (imp/crew)' with total padded to align parens, then whole thing padded to col_width."""
    inner = f"{total:<{total_width}} ({Colors.RED}{imp}{Colors.RESET}/{Colors.GREEN}{crew}{Colors.RESET})"
    return pad_right(inner, col_width)


def print_rankings(stats: dict[str, dict], games_analyzed: int) -> list[dict]:
    """Print formatted rankings table to terminal and return rankings data for JSON."""
    # Sort by overall ELO (descending), then by impostor ELO
    sorted_players = sorted(
        stats.items(),
        key=lambda x: (x[1]["overall_elo"] or 0, x[1]["impostor_elo"] or 0),
        reverse=True
    )

    print()
    print(f"    {Colors.CYAN}{Colors.BOLD}Rankings{Colors.RESET} {Colors.CYAN}({games_analyzed} games){Colors.RESET}")
    print(f"    ({Colors.RED}impostor{Colors.RESET}/{Colors.GREEN}crew{Colors.RESET})")
    print()

    # Header with compact format - ELO/Won/Games/Survived show (imp/crew) inline
    print(f"    {Colors.CYAN}{'#':<3} {'LLM':<24} {'ELO':<18} {'Won':<20} {'Games':<12} {'Survived':<20} {'Sacrificed':<11} {'Killed':<8} {'Accuracy':<9} {'Messages':<9}{Colors.RESET}")
    print(f"    {'─' * 143}")

    rankings = []

    for rank, (name, player_stats) in enumerate(sorted_players, 1):
        total_games = player_stats['games_played']
        imp_games = player_stats['impostor_games']
        crew_games = player_stats['crew_games']

        # Basic stats - total_width=2 for games count, col_width=12
        games = format_with_roles(
            str(total_games),
            str(imp_games),
            str(crew_games),
            2, 12
        )

        # Overall stats
        overall = str(player_stats["overall_elo"]) if player_stats["overall_elo"] else "-"
        win_pct = f"{player_stats['win_rate']*100:.0f}%"
        survived = f"{player_stats['survival_rate']*100:.0f}%"

        # Impostor stats
        imp_elo = str(player_stats["impostor_elo"]) if imp_games > 0 else "-"
        imp_wr = f"{player_stats['impostor_win_rate']*100:.0f}%" if player_stats['impostor_win_rate'] is not None else "-"
        imp_surv = f"{player_stats['impostor_survival_rate']*100:.0f}%" if player_stats['impostor_survival_rate'] is not None else "-"

        # Crew stats
        crew_elo = str(player_stats["crew_elo"]) if crew_games > 0 else "-"
        crew_wr = f"{player_stats['crew_win_rate']*100:.0f}%" if player_stats['crew_win_rate'] is not None else "-"
        crew_surv = f"{player_stats['crew_survival_rate']*100:.0f}%" if player_stats['crew_survival_rate'] is not None else "-"

        # Format combined columns with role breakdown
        # total_width: 4 for ELO (4 digits), 4 for percentages (100%)
        elo_col = format_with_roles(overall, imp_elo, crew_elo, 4, 18)
        won_col = format_with_roles(win_pct, imp_wr, crew_wr, 4, 20)
        surv_col = format_with_roles(survived, imp_surv, crew_surv, 4, 20)

        # Other stats
        sacrificed_pct = player_stats['times_voted_out'] / total_games if total_games > 0 else None
        killed_pct = player_stats['times_killed'] / total_games if crew_games > 0 else None
        accuracy = player_stats['voting_accuracy']
        messages = round(player_stats['avg_messages_per_day'], 1)

        # Display formatting
        voted_out_str = f"{sacrificed_pct*100:.0f}%" if sacrificed_pct is not None else "-"
        killed_str = f"{killed_pct*100:.0f}%" if killed_pct is not None else "-"
        vote_acc_str = f"{accuracy*100:.0f}%" if accuracy is not None else "-"
        msg_per_day_str = f"{messages:.1f}"

        print(f"    {rank:<3} {name:<24} {elo_col} {won_col} {games} {surv_col} {voted_out_str:<11} {killed_str:<8} {vote_acc_str:<9} {msg_per_day_str:<9}")

        # Build JSON data (mirrors console output)
        rankings.append({
            "rank": rank,
            "llm": name,
            "elo": {
                "overall": player_stats["overall_elo"],
                "impostor": player_stats["impostor_elo"] if imp_games > 0 else None,
                "crew": player_stats["crew_elo"] if crew_games > 0 else None
            },
            "won": {
                "overall": round(player_stats['win_rate'], 2),
                "impostor": round(player_stats['impostor_win_rate'], 2) if player_stats['impostor_win_rate'] is not None else None,
                "crew": round(player_stats['crew_win_rate'], 2) if player_stats['crew_win_rate'] is not None else None
            },
            "games": {
                "total": total_games,
                "impostor": imp_games,
                "crew": crew_games
            },
            "survived": {
                "overall": round(player_stats['survival_rate'], 2),
                "impostor": round(player_stats['impostor_survival_rate'], 2) if player_stats['impostor_survival_rate'] is not None else None,
                "crew": round(player_stats['crew_survival_rate'], 2) if player_stats['crew_survival_rate'] is not None else None
            },
            "sacrificed": round(sacrificed_pct, 2) if sacrificed_pct is not None else None,
            "killed": round(killed_pct, 2) if killed_pct is not None else None,
            "accuracy": round(accuracy, 2) if accuracy is not None else None,
            "messages": messages
        })

    print()
    return rankings


def save_json(rankings: list[dict], games_analyzed: int, output_path: str = "stats.json") -> None:
    """Save statistics to JSON file (mirrors console output)."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "games_analyzed": games_analyzed,
        "rankings": rankings
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{Colors.YELLOW}Stats saved to {output_path}{Colors.RESET}")


def main():
    """Main entry point."""
    # Load games
    games = load_games()

    if not games:
        print("No completed games found in games/ directory")
        return

    print(f"{Colors.YELLOW}Found {len(games)} completed games{Colors.RESET}")

    # Initialize player tracking
    players = {}

    # Process each game chronologically
    for game in games:
        process_game(game, players)

    # Calculate derived statistics
    stats = calculate_derived_stats(players)

    # Output results
    rankings = print_rankings(stats, len(games))
    save_json(rankings, len(games))


if __name__ == "__main__":
    main()
