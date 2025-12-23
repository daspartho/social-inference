#!/usr/bin/env python3
"""
Calculates role-based ELO ratings and statistics from game transcripts.

ELO System:
- Team vs Team: Your team's average ELO vs opponent team's average ELO
- Everyone on the same team gets the same rating change
- Two separate ratings: impostor_elo and crew_elo
- Overall ELO is simple average of impostor and crew ELO (if both played)
"""

import json
import os
import glob
import re
import argparse
from datetime import datetime

STARTING_ELO = 1000
BASE_K_FACTOR = 32


def safe_div(num: float, denom: float, default=None) -> float | None:
    """Safely divide, returning default if denominator is 0."""
    return num / denom if denom else default


def pct(value: float | None) -> str:
    """Format a rate as percentage string, or '-' if None."""
    return f"{value*100:.0f}%" if value is not None else "-"


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
    game_files = [f for f in game_files if "_partial" not in f]

    games = []
    for filepath in game_files:
        with open(filepath, "r") as f:
            games.append(json.load(f))

    games.sort(key=lambda g: g.get("timestamp", ""))
    return games


def initialize_player_stats() -> dict:
    return {
        # ELO
        "impostor_elo": STARTING_ELO,
        "crew_elo": STARTING_ELO,
        # Games
        "impostor_games": 0,
        "impostor_wins": 0,
        "crew_games": 0,
        "crew_wins": 0,
        # Survival
        "impostor_survived": 0,
        "crew_survived": 0,
        "times_voted_out": 0,
        "times_killed": 0,
        # Voting accuracy (crew only - impostors know who to vote for)
        "votes_cast_as_crew": 0,
        "votes_for_impostors": 0,
        # Activity
        "messages_as_impostor": 0,
        "messages_as_crew": 0,
        "turns_as_impostor": 0,
        "turns_as_crew": 0,
        "asks_as_impostor": 0,
        "asks_as_crew": 0,
        "asked_as_impostor": 0,
        "asked_as_crew": 0,
        # Majority vote tracking
        "majority_votes_as_impostor": 0,
        "majority_votes_as_crew": 0,
        "final_votes_as_impostor": 0,
        "final_votes_as_crew": 0,
        # Detective
        "detective_investigations": 0,
        "detective_catches": 0,
        "times_investigated": 0,
        "times_caught": 0,
    }


def get_elimination_info(game: dict, player_name: str) -> tuple[int, str] | None:
    """Get elimination day and type for a player, or None if survived."""
    for elim in game.get("elimination_order", []):
        if elim["name"] == player_name:
            return elim["day"], elim["type"]
    return None


def process_votes(votes: dict, players: dict[str, dict], impostors: set) -> None:
    """Process votes from a single vote round for accuracy tracking."""
    for target, voters in votes.items():
        for voter in voters:
            if voter not in players or voter in impostors:
                continue
            players[voter]["votes_cast_as_crew"] += 1
            if target in impostors:
                players[voter]["votes_for_impostors"] += 1


def process_game(game: dict, players: dict[str, dict]) -> None:
    winner = game["winner"]
    impostors = set(game["impostors"])
    all_players = [p["name"] for p in game["players"]]
    crew_members = [p for p in all_players if p not in impostors]

    for player in all_players:
        if player not in players:
            players[player] = initialize_player_stats()

    impostor_won = (winner == "impostors")
    impostor_elos = {p: players[p]["impostor_elo"] for p in impostors}
    crew_elos = {p: players[p]["crew_elo"] for p in crew_members}

    avg_impostor_elo = sum(impostor_elos.values()) / len(impostor_elos) if impostor_elos else STARTING_ELO
    avg_crew_elo = sum(crew_elos.values()) / len(crew_elos) if crew_elos else STARTING_ELO

    impostor_change = calculate_elo_change(avg_impostor_elo, avg_crew_elo, impostor_won, BASE_K_FACTOR)
    crew_change = calculate_elo_change(avg_crew_elo, avg_impostor_elo, not impostor_won, BASE_K_FACTOR)

    for imp in impostors:
        players[imp]["impostor_elo"] += impostor_change
    for crew in crew_members:
        players[crew]["crew_elo"] += crew_change

    for player_name in all_players:
        stats = players[player_name]
        is_impostor = player_name in impostors
        player_won = (is_impostor and impostor_won) or (not is_impostor and not impostor_won)

        if is_impostor:
            stats["impostor_games"] += 1
            if player_won:
                stats["impostor_wins"] += 1
        else:
            stats["crew_games"] += 1
            if player_won:
                stats["crew_wins"] += 1

        elim_info = get_elimination_info(game, player_name)
        if elim_info is None:
            if is_impostor:
                stats["impostor_survived"] += 1
            else:
                stats["crew_survived"] += 1
        else:
            _, elim_type = elim_info
            if elim_type == "voted":
                stats["times_voted_out"] += 1
            elif elim_type == "killed":
                stats["times_killed"] += 1

    for round_data in game.get("rounds", []):
        for msg in round_data.get("discussion", []):
            speaker = msg["player"]
            if speaker not in players:
                continue

            stats = players[speaker]
            role = "impostor" if speaker in impostors else "crew"

            stats[f"turns_as_{role}"] += 1
            if msg.get("action") != "pass":
                stats[f"messages_as_{role}"] += 1
                if re.search(r'/\s*ask\b', msg.get("message", ""), re.IGNORECASE):
                    stats[f"asks_as_{role}"] += 1
            if msg.get("asked_by"):
                stats[f"asked_as_{role}"] += 1

        vote_rounds = round_data.get("vote_rounds", [])
        for vote_round in vote_rounds:
            process_votes(vote_round.get("votes", {}), players, impostors)

        sacrificed = round_data.get("vote_result", {}).get("sacrificed")
        if sacrificed and vote_rounds:
            final_votes = vote_rounds[-1].get("votes", {})
            for target, voters in final_votes.items():
                for voter in voters:
                    if voter not in players:
                        continue
                    is_impostor = voter in impostors
                    if is_impostor:
                        players[voter]["final_votes_as_impostor"] += 1
                        if target == sacrificed:
                            players[voter]["majority_votes_as_impostor"] += 1
                    else:
                        players[voter]["final_votes_as_crew"] += 1
                        if target == sacrificed:
                            players[voter]["majority_votes_as_crew"] += 1

        night = round_data.get("night") or {}
        investigation = night.get("investigation")
        if investigation:
            detective = investigation.get("investigator")
            if detective and detective in players:
                players[detective]["detective_investigations"] += 1
                if investigation.get("result") == "impostor":
                    players[detective]["detective_catches"] += 1
            target = investigation.get("target")
            if target and target in players:
                players[target]["times_investigated"] += 1
                if investigation.get("result") == "impostor":
                    players[target]["times_caught"] += 1


def calculate_derived_stats(players: dict[str, dict]) -> dict[str, dict]:
    results = {}

    for name, s in players.items():
        imp_games = s["impostor_games"]
        crew_games = s["crew_games"]
        games = imp_games + crew_games

        total_messages = s["messages_as_impostor"] + s["messages_as_crew"]
        total_turns = s["turns_as_impostor"] + s["turns_as_crew"]
        total_passes = total_turns - total_messages

        results[name] = {
            "impostor_elo": round(s["impostor_elo"]),
            "crew_elo": round(s["crew_elo"]),
            "overall_elo": None,
            "games_played": games,
            "impostor_games": imp_games,
            "crew_games": crew_games,
            "win_rate": safe_div(s["impostor_wins"] + s["crew_wins"], games, 0),
            "impostor_win_rate": safe_div(s["impostor_wins"], imp_games),
            "crew_win_rate": safe_div(s["crew_wins"], crew_games),
            "survival_rate": safe_div(s["impostor_survived"] + s["crew_survived"], games, 0),
            "impostor_survival_rate": safe_div(s["impostor_survived"], imp_games),
            "crew_survival_rate": safe_div(s["crew_survived"], crew_games),
            "sacrificed_rate": safe_div(s["times_voted_out"], games),
            "killed_rate": safe_div(s["times_killed"], crew_games),
            "voting_accuracy": safe_div(s["votes_for_impostors"], s["votes_cast_as_crew"]),
            "majority_rate": safe_div(
                s["majority_votes_as_impostor"] + s["majority_votes_as_crew"],
                s["final_votes_as_impostor"] + s["final_votes_as_crew"]
            ),
            "majority_rate_impostor": safe_div(s["majority_votes_as_impostor"], s["final_votes_as_impostor"]),
            "majority_rate_crew": safe_div(s["majority_votes_as_crew"], s["final_votes_as_crew"]),
            "ask_rate": safe_div(s["asks_as_impostor"] + s["asks_as_crew"], total_messages),
            "ask_rate_impostor": safe_div(s["asks_as_impostor"], s["messages_as_impostor"]),
            "ask_rate_crew": safe_div(s["asks_as_crew"], s["messages_as_crew"]),
            "asked_rate": safe_div(s["asked_as_impostor"] + s["asked_as_crew"], total_turns),
            "asked_rate_impostor": safe_div(s["asked_as_impostor"], s["turns_as_impostor"]),
            "asked_rate_crew": safe_div(s["asked_as_crew"], s["turns_as_crew"]),
            "pass_rate": safe_div(total_passes, total_turns),
            "detective_investigations": s["detective_investigations"],
            "detective_catches": s["detective_catches"],
            "detective_catch_rate": safe_div(s["detective_catches"], s["detective_investigations"]),
            "times_investigated": s["times_investigated"],
            "times_caught": s["times_caught"],
            "caught_rate": safe_div(s["times_caught"], s["times_investigated"]),
        }

        r = results[name]
        if imp_games > 0 and crew_games > 0:
            r["overall_elo"] = round((s["impostor_elo"] + s["crew_elo"]) / 2)
        elif imp_games > 0:
            r["overall_elo"] = round(s["impostor_elo"])
        elif crew_games > 0:
            r["overall_elo"] = round(s["crew_elo"])

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


def format_with_roles(total: str, imp: str, crew: str, total_width: int, col_width: int) -> str:
    """Format as 'total (imp/crew)' with padding."""
    inner = f"{total:<{total_width}} ({Colors.RED}{imp}{Colors.RESET}/{Colors.GREEN}{crew}{Colors.RESET})"
    return inner + ' ' * (col_width - visual_len(inner))


def rate_obj(overall: float | None, imp: float | None, crew: float | None) -> dict:
    """Build a rate object with overall/impostor/crew breakdown."""
    return {
        "overall": round(overall, 2) if overall is not None else None,
        "impostor": round(imp, 2) if imp is not None else None,
        "crew": round(crew, 2) if crew is not None else None
    }


def print_rankings(stats: dict[str, dict], games_analyzed: int) -> list[dict]:
    """Print formatted rankings table to terminal and return rankings data for JSON."""
    sorted_players = sorted(
        stats.items(),
        key=lambda x: (x[1]["overall_elo"] or 0, x[1]["impostor_elo"] or 0),
        reverse=True
    )

    print()
    print(f"    {Colors.CYAN}{Colors.BOLD}Rankings{Colors.RESET} {Colors.CYAN}({games_analyzed} games){Colors.RESET}")
    print(f"    ({Colors.RED}impostor{Colors.RESET}/{Colors.GREEN}crew{Colors.RESET})")
    print()
    print(f"    {Colors.CYAN}{'#':<3} {'LLM':<24} {'ELO':<18} {'Won':<20} {'Games':<12} {'Survived':<20} {'Sacrificed':<11} {'Killed':<8} {'Accuracy':<9} {'Majority':<18} {'Asks':<18} {'Asked':<18} {'Detects':<10} {'Detected':<10} {'Pass':<6}{Colors.RESET}")
    print(f"    {'─' * 218}")

    rankings = []

    for rank, (name, p) in enumerate(sorted_players, 1):
        imp_games = p['impostor_games']
        crew_games = p['crew_games']
        total_games = p['games_played']

        games_col = format_with_roles(str(total_games), str(imp_games), str(crew_games), 2, 12)

        elo_col = format_with_roles(
            str(p["overall_elo"]) if p["overall_elo"] else "-",
            str(p["impostor_elo"]) if imp_games > 0 else "-",
            str(p["crew_elo"]) if crew_games > 0 else "-",
            4, 18
        )

        won_col = format_with_roles(
            pct(p['win_rate']),
            pct(p['impostor_win_rate']),
            pct(p['crew_win_rate']),
            4, 20
        )

        surv_col = format_with_roles(
            pct(p['survival_rate']),
            pct(p['impostor_survival_rate']),
            pct(p['crew_survival_rate']),
            4, 20
        )

        ask_col = format_with_roles(pct(p['ask_rate']), pct(p['ask_rate_impostor']), pct(p['ask_rate_crew']), 4, 18)
        asked_col = format_with_roles(pct(p['asked_rate']), pct(p['asked_rate_impostor']), pct(p['asked_rate_crew']), 4, 18)
        maj_col = format_with_roles(pct(p['majority_rate']), pct(p['majority_rate_impostor']), pct(p['majority_rate_crew']), 4, 18)

        sacrificed = pct(p['sacrificed_rate'])
        killed = pct(p['killed_rate'])

        det_rate = p['detective_catch_rate']
        det_inv = p['detective_investigations']
        det_str = f"{pct(det_rate)} ({det_inv})" if det_inv > 0 else "-"

        caught_rate = p['caught_rate']
        inv_count = p['times_investigated']
        detected_str = f"{pct(caught_rate)} ({inv_count})" if inv_count > 0 else "-"

        print(f"    {rank:<3} {name:<24} {elo_col} {won_col} {games_col} {surv_col} {sacrificed:<11} {killed:<8} {pct(p['voting_accuracy']):<9} {maj_col} {ask_col} {asked_col} {det_str:<10} {detected_str:<10} {pct(p['pass_rate']):<6}")

        rankings.append({
            "rank": rank,
            "llm": name,
            "elo": {"overall": p["overall_elo"], "impostor": p["impostor_elo"] if imp_games > 0 else None, "crew": p["crew_elo"] if crew_games > 0 else None},
            "won": rate_obj(p['win_rate'], p['impostor_win_rate'], p['crew_win_rate']),
            "games": {"total": total_games, "impostor": imp_games, "crew": crew_games},
            "survived": rate_obj(p['survival_rate'], p['impostor_survival_rate'], p['crew_survival_rate']),
            "sacrificed": round(p['sacrificed_rate'], 2) if p['sacrificed_rate'] is not None else None,
            "killed": round(p['killed_rate'], 2) if p['killed_rate'] is not None else None,
            "accuracy": round(p['voting_accuracy'], 2) if p['voting_accuracy'] is not None else None,
            "majority_rate": rate_obj(p['majority_rate'], p['majority_rate_impostor'], p['majority_rate_crew']),
            "ask_rate": rate_obj(p['ask_rate'], p['ask_rate_impostor'], p['ask_rate_crew']),
            "asked_rate": rate_obj(p['asked_rate'], p['asked_rate_impostor'], p['asked_rate_crew']),
            "pass_rate": round(p['pass_rate'], 2) if p['pass_rate'] is not None else None,
            "detective": {
                "investigations": p['detective_investigations'],
                "catches": p['detective_catches'],
                "catch_rate": round(det_rate, 2) if det_rate is not None else None
            },
            "detected": {
                "investigations": p['times_investigated'],
                "caught": p['times_caught'],
                "caught_rate": round(caught_rate, 2) if caught_rate is not None else None
            }
        })

    print()
    return rankings


def save_json(rankings: list[dict], games_analyzed: int, output_path: str = "stats.json") -> None:
    """Save statistics to JSON file."""
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "games_analyzed": games_analyzed,
            "rankings": rankings
        }, f, indent=2)
    print(f"{Colors.YELLOW}Stats saved to {output_path}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Calculate ELO & stats from game transcripts")
    parser.add_argument("--last", "-n", type=int, help="Only analyze the last N games")
    args = parser.parse_args()

    games = load_games()
    if not games:
        print("No completed games found in games/ directory")
        return

    if args.last:
        games = games[-args.last:]

    print(f"{Colors.YELLOW}Analyzing {len(games)} games{Colors.RESET}")

    players = {}
    for game in games:
        process_game(game, players)

    stats = calculate_derived_stats(players)
    rankings = print_rankings(stats, len(games))
    save_json(rankings, len(games))


if __name__ == "__main__":
    main()
