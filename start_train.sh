#!/bin/bash

# Function to list all tmux sessions
list_sessions() {
    tmux list-sessions 2>/dev/null || echo "No tmux sessions found."
}

# Function to start or attach to a session
start_session() {
    local SESSION_NAME=$1
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Attaching to existing session: $SESSION_NAME"
        tmux attach -t "$SESSION_NAME"
    else
        echo "Starting new session: $SESSION_NAME"
        tmux new-session -s "$SESSION_NAME"
    fi
}

# Function to delete a specific session
delete_session() {
    local SESSION_NAME=$1
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Deleting session: $SESSION_NAME"
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Session '$SESSION_NAME' not found."
    fi
}

# Function to kill all tmux sessions
delete_all_sessions() {
    if tmux list-sessions 2>/dev/null; then
        echo "Killing all tmux sessions..."
        tmux kill-server
    else
        echo "No tmux sessions to kill."
    fi
}

# Function to check if tmux is installed
check_tmux_installed() {
    if ! command -v tmux &>/dev/null; then
        echo "Error: tmux is not installed. Please install it using your package manager."
        exit 1
    fi
}

# Ensure tmux is installed
check_tmux_installed

# Display menu
while true; do
    echo "Tmux Manager:"
    echo "1) List sessions"
    echo "2) Start or attach to session"
    echo "3) Delete a specific session"
    echo "4) Kill all sessions"
    echo "5) Exit"
    
    read -rp "Choose an option (1-5): " OPTION

    case $OPTION in
        1)
            list_sessions
            ;;
        2)
            read -rp "Enter session name: " SESSION_NAME
            start_session "$SESSION_NAME"
            ;;
        3)
            read -rp "Enter session name to delete: " SESSION_NAME
            delete_session "$SESSION_NAME"
            ;;
        4)
            delete_all_sessions
            ;;
        5)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please choose a number between 1 and 5."
            ;;
    esac
    echo ""
done