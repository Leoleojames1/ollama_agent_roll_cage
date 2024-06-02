```mermaid
stateDiagram
    [*] --> Game_Over
    Game_Over --> Game_Running: Player Clicks "Start"
    Game_Running --> Game_Paused: Player Clicks "Pause"
    Game_Paused --> Game_Running: Player Clicks "Resume"
    Game_Running --> Game_Over: Player Dies