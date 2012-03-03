/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_tty_h_
#define _lua_tty_h_

#include "plm_config.h"

#include <stdio.h>

/* option for multiple returns in `lua_pcall' and `lua_call' */
#define TTY_MULTRET	(-1)
#define TTY_QL(x)	LUA_QL(x)
#define TTY_PROGNAME		"plastimatch"
#define TTY_MAXINPUT 255
#define TTY_PROMPT "% "
#define TTY_PROMPT2 "> "


#if defined __cplusplus
extern "C" {
#endif

/* For TTY history */
#if (READLINE_FOUND)
#include <readline/readline.h>
#include <readline/history.h>
#define lua_readline(L,b,p)	((void)L, ((b)=readline(p)) != NULL)
#define lua_saveline(L,idx) \
	if (lua_strlen(L,idx) > 0)  /* non-empty line? */ \
	  add_history(lua_tostring(L, idx));  /* add it to history */
#define lua_freeline(L,b)	((void)L, free(b))
#else
#define lua_readline(L,b,p)	\
	((void)L, fputs(p, stdout), fflush(stdout),  /* show prompt */ \
	fgets(b, TTY_MAXINPUT, stdin) != NULL)  /* get line */
#define lua_saveline(L,idx)	{ (void)L; (void)idx; }
#define lua_freeline(L,b)	{ (void)L; (void)b; }
#endif

void do_tty (lua_State *L);
void do_stdin (lua_State *L);

#if defined __cplusplus
}
#endif



#endif
