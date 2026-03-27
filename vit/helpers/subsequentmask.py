import torch

def subsequent_mask(size):
	subsequent_mask = torch.ones(1, size, size).tril().bool()
	return subsequent_mask == 0

if __name__ == "__main__":
	import pandas as pd
	import altair as alt

	SIZE = 20

	LS_data = pd.concat(
		[
			pd.DataFrame(
				{
					"Subsequent Mask": subsequent_mask(SIZE)[0][x, y].flatten(),
					"Window": x,
					"Masking": y
					
				}
			)
			for x in range(SIZE)
			for y in range(SIZE)
		]
	)

	alt.Chart(LS_data).mark_rect().properties(
		height=250, width=250
	).encode(
		alt.X("Window:O"),
    alt.Y("Masking:O"),
    alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
  ).interactive().save("html/transformer_mask.html")