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
from datetime import datetime

# ELO Parameters
STARTING_ELO = 1000
BASE_K_FACTOR = 32


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


def get_player_role(game: dict, player_name: str) -> str:
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
        result = {
            # ELO ratings
            "impostor_elo": round(stats["impostor_elo"]),
            "crew_elo": round(stats["crew_elo"]),

            # Overall ELO (weighted average)
            "overall_elo": None,

            # Game counts
            "games_played": stats["games_played"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": stats["wins"] / stats["games_played"] if stats["games_played"] > 0 else 0,

            # Role breakdown
            "impostor_games": stats["impostor_games"],
            "impostor_wins": stats["impostor_wins"],
            "impostor_win_rate": stats["impostor_wins"] / stats["impostor_games"] if stats["impostor_games"] > 0 else None,
            "crew_games": stats["crew_games"],
            "crew_wins": stats["crew_wins"],
            "crew_win_rate": stats["crew_wins"] / stats["crew_games"] if stats["crew_games"] > 0 else None,

            # Survival stats
            "survival_rate": stats["games_survived"] / stats["games_played"] if stats["games_played"] > 0 else 0,
            "avg_days_survived": stats["total_days_survived"] / stats["games_played"] if stats["games_played"] > 0 else 0,
            "times_voted_out": stats["times_voted_out"],
            "times_killed": stats["times_killed"],

            # Voting accuracy
            "votes_cast_as_crew": stats["votes_cast_as_crew"],
            "votes_for_impostors": stats["votes_for_impostors"],
            "voting_accuracy": stats["votes_for_impostors"] / stats["votes_cast_as_crew"] if stats["votes_cast_as_crew"] > 0 else None,

            # Vote reception
            "times_received_votes": stats["times_received_votes"],
            "avg_votes_against_per_game": stats["times_received_votes"] / stats["games_played"] if stats["games_played"] > 0 else 0,

            # Activity
            "total_messages": stats["total_messages"],
            "avg_messages_per_game": stats["total_messages"] / stats["games_played"] if stats["games_played"] > 0 else 0,
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


def print_rankings(stats: dict[str, dict], games_analyzed: int) -> None:
    """Print formatted rankings table to terminal."""
    # Sort by overall ELO (descending), then by impostor ELO
    sorted_players = sorted(
        stats.items(),
        key=lambda x: (x[1]["overall_elo"] or 0, x[1]["impostor_elo"] or 0),
        reverse=True
    )

    # Header
    print()
    print("=" * 95)
    print(f"{'LLM SOCIAL DEDUCTION RANKINGS':^95}")
    print(f"{'Games Analyzed: ' + str(games_analyzed):^95}")
    print("=" * 95)
    print()

    # Rankings table
    header = f"{'Rank':<5} {'Player':<25} {'Overall':>8} {'Imp ELO':>8} {'Crew ELO':>9} {'Win%':>6} {'Games':>12}"
    print(header)
    print("-" * 95)

    for rank, (name, player_stats) in enumerate(sorted_players, 1):
        overall = str(player_stats["overall_elo"]) if player_stats["overall_elo"] else "-"
        imp_elo = str(player_stats["impostor_elo"]) if player_stats["impostor_games"] > 0 else "-"
        crew_elo = str(player_stats["crew_elo"]) if player_stats["crew_games"] > 0 else "-"
        win_pct = f"{player_stats['win_rate']*100:.0f}%"
        games = f"{player_stats['games_played']} ({player_stats['impostor_games']}I/{player_stats['crew_games']}C)"

        print(f"{rank:<5} {name:<25} {overall:>8} {imp_elo:>8} {crew_elo:>9} {win_pct:>6} {games:>12}")

    print("-" * 95)
    print()

    # Detailed stats for each player
    print("=" * 95)
    print(f"{'DETAILED PLAYER STATISTICS':^95}")
    print("=" * 95)

    for name, player_stats in sorted_players:
        print()
        print(f"┌{'─' * 93}┐")
        print(f"│ {name:<92}│")
        print(f"├{'─' * 93}┤")

        # Role stats
        imp_wr = f"{player_stats['impostor_win_rate']*100:.0f}%" if player_stats['impostor_win_rate'] is not None else "N/A"
        crew_wr = f"{player_stats['crew_win_rate']*100:.0f}%" if player_stats['crew_win_rate'] is not None else "N/A"
        imp_elo_display = player_stats['impostor_elo'] if player_stats['impostor_games'] > 0 else "N/A"
        crew_elo_display = player_stats['crew_elo'] if player_stats['crew_games'] > 0 else "N/A"

        line1 = f"│ Impostor: {player_stats['impostor_games']} games, {imp_wr} win rate, ELO {imp_elo_display}"
        print(f"{line1:<94}│")

        line2 = f"│ Crew: {player_stats['crew_games']} games, {crew_wr} win rate, ELO {crew_elo_display}"
        print(f"{line2:<94}│")

        # Survival and voting
        surv = f"{player_stats['survival_rate']*100:.0f}%"
        vote_acc = f"{player_stats['voting_accuracy']*100:.0f}%" if player_stats['voting_accuracy'] is not None else "N/A"

        line3 = f"│ Survival: {surv} | Avg days: {player_stats['avg_days_survived']:.1f} | Voted out: {player_stats['times_voted_out']} | Killed: {player_stats['times_killed']}"
        print(f"{line3:<94}│")

        line4 = f"│ Voting accuracy (as crew): {vote_acc} | Messages/game: {player_stats['avg_messages_per_game']:.1f} | Votes against: {player_stats['times_received_votes']}"
        print(f"{line4:<94}│")

        print(f"└{'─' * 93}┘")

    print()


def save_json(stats: dict[str, dict], games_analyzed: int, output_path: str = "stats.json") -> None:
    """Save statistics to JSON file."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "games_analyzed": games_analyzed,
        "players": stats
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Stats saved to {output_path}")


def main():
    """Main entry point."""
    # Load games
    games = load_games()

    if not games:
        print("No completed games found in games/ directory")
        return

    print(f"Found {len(games)} completed games")

    # Initialize player tracking
    players = {}

    # Process each game chronologically
    for game in games:
        process_game(game, players)

    # Calculate derived statistics
    stats = calculate_derived_stats(players)

    # Output results
    print_rankings(stats, len(games))
    save_json(stats, len(games))


if __name__ == "__main__":
    main()
