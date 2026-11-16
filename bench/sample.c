/*
 * sample.c - a small elevator controller, written for cranker-llm's benchmarks.
 * Its functions are what the map test lists, elevator_step's callers what the agent test finds, and the local
 * variable 'state' (never the field ->state) what the rename test renames.
 */

#include <stdint.h>
#include <string.h>

#define FLOORS 12
#define DOOR_FRAMES 90
#define QUEUE_SIZE 16

enum { IDLE, MOVING_UP, MOVING_DOWN, DOOR_OPEN, OUT_OF_SERVICE };

struct elevator {
	uint8_t state;
	uint8_t floor;
	uint8_t target;
	uint16_t door_timer;
	uint8_t queue[QUEUE_SIZE];
	uint8_t head, tail;
	uint32_t trips;
};

static int queue_empty(const struct elevator *e)
{
	return e->head == e->tail;
}

static void queue_push(struct elevator *e, uint8_t floor)
{
	uint8_t next = (uint8_t)((e->tail + 1) % QUEUE_SIZE);
	if (next == e->head || floor >= FLOORS)
		return; /* full, or no such floor: the call is dropped */
	for (uint8_t i = e->head; i != e->tail; i = (uint8_t)((i + 1) % QUEUE_SIZE))
		if (e->queue[i] == floor)
			return; /* already called */
	e->queue[e->tail] = floor;
	e->tail = next;
}

static uint8_t queue_pop(struct elevator *e)
{
	uint8_t floor = e->queue[e->head];
	e->head = (uint8_t)((e->head + 1) % QUEUE_SIZE);
	return floor;
}

void elevator_init(struct elevator *e)
{
	memset(e, 0, sizeof *e);
	e->state = IDLE;
}

void elevator_call(struct elevator *e, uint8_t floor)
{
	if (e->state == OUT_OF_SERVICE)
		return;
	queue_push(e, floor);
}

static uint8_t next_state(const struct elevator *e)
{
	uint8_t state = e->state;

	if (state == OUT_OF_SERVICE || state == DOOR_OPEN)
		return state;
	if (e->floor == e->target)
		state = queue_empty(e) ? IDLE : DOOR_OPEN;
	else
		state = e->floor < e->target ? MOVING_UP : MOVING_DOWN;
	return state;
}

void elevator_step(struct elevator *e)
{
	uint8_t state = next_state(e);

	switch (state) {
	case IDLE:
		if (!queue_empty(e))
			e->target = queue_pop(e);
		break;
	case MOVING_UP:
		e->floor++;
		break;
	case MOVING_DOWN:
		e->floor--;
		break;
	case DOOR_OPEN:
		if (e->door_timer == 0) {
			e->door_timer = DOOR_FRAMES;
		} else if (--e->door_timer == 0) {
			e->trips++;
			state = IDLE;
			if (!queue_empty(e))
				e->target = queue_pop(e);
		}
		break;
	}
	e->state = state;
}

void elevator_run(struct elevator *e, int frames)
{
	while (frames-- > 0)
		elevator_step(e);
}

void elevator_fault(struct elevator *e)
{
	e->state = OUT_OF_SERVICE;
	e->head = e->tail = 0;
	elevator_step(e);
}
