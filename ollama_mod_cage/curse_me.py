""" curse_me.py
    a file for rendering multiple processes in the command line utilizing the curses library

"""
import curses

def draw_menu(stdscr):
    k = 0
    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()
    while (k != ord('q')):
        # Initialization
        height, width = stdscr.getmaxyx()

        cols_tot = width
        rows_tot = height
        cols_mid = int(0.5*cols_tot)  # middle point of the window
        rows_mid = int(0.5*rows_tot)

        pad11 = curses.newpad(rows_mid, cols_mid)
        pad12 = curses.newpad(rows_mid, cols_mid)
        pad21 = curses.newpad(rows_mid, cols_mid)
        pad22 = curses.newpad(rows_mid, cols_mid)
        pad11.addstr(0, 0, "*** PROCESS 01 ***")
        pad12.addstr(0, 0, "*** PROCESS 02 ***")
        pad21.addstr(0, 0, "*** PROCESS 03 ***")
        pad22.addstr(0, 0, "*** PROCESS 04 ***")
        pad11.refresh(0, 0, 0, 0, rows_mid, cols_mid)
        pad12.refresh(0, 0, 0, cols_mid, rows_mid, cols_tot-1)
        pad21.refresh(0, 0, rows_mid, 0, rows_mid, cols_mid)
        pad22.refresh(0, 0, rows_mid, cols_mid, rows_tot-1, cols_tot-1)

        k = stdscr.getch()

def main():
    curses.wrapper(draw_menu)

if __name__ == "__main__":
    main()