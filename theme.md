# Input controls

## Scrubbable numeric field with stepper buttons

Used by decal transforms and other values that benefit from small, deliberate
steps. This is a Blender-style value control made from three adjoining parts:

- A `−` button that decreases the value by one configured step.
- A central numeric field.
- A `+` button that increases the value by one configured step.

The central field supports horizontal scrubbing: dragging left decreases the
value and dragging right increases it. Clicking the number enables exact inline
entry and highlights the current value for replacement. Pressing Enter or
clicking away finishes the edit. Cmd-clicking restores the default value.

Use this control when discrete nudging is important in addition to fast dragging
and exact entry, such as position, rotation, scale, depth, and modifier values.

## Scrub input

Used by procedural-generator parameters. It is a compact numeric field without
separate `−` and `+` buttons.

Dragging horizontally adjusts the value continuously. The cursor changes to a
horizontal-resize cursor to indicate that the field can be scrubbed. Double-
clicking replaces the field in place with an exact numeric editor and highlights
the current value. Pressing Enter or clicking away finishes the edit.
Cmd-clicking restores the parameter's generator-specific default.

Use this control for dense parameter groups where fast continuous exploration is
more useful than single-step buttons, such as noise scale, distortion, scratch
width, mortar thickness, and procedural detail.

