// Distance from the bottom, in pixels, within which the viewport still counts as
// following the content. Absorbs sub-pixel rounding and small layout shifts.
const PINNED_THRESHOLD = 48;

export interface StickToBottomOptions {
	root: HTMLElement | null;
}

export function stickToBottom(node: HTMLElement, initialOptions: StickToBottomOptions) {
	let root = initialOptions.root;
	let pinned = true;
	const observer = new ResizeObserver(() => {
		if (pinned && root) root.scrollTop = root.scrollHeight;
	});

	// Content growth never fires `scroll`, so only user or programmatic scrolls
	// update the pin; scrolling up releases it and returning to the bottom restores it.
	function trackPin(): void {
		if (!root) return;
		pinned = root.scrollHeight - root.scrollTop - root.clientHeight <= PINNED_THRESHOLD;
	}

	function attach(): void {
		if (!root) return;
		root.addEventListener('scroll', trackPin, { passive: true });
		observer.observe(root);
		root.scrollTop = root.scrollHeight;
	}

	function detach(): void {
		if (!root) return;
		root.removeEventListener('scroll', trackPin);
		observer.unobserve(root);
	}

	observer.observe(node);
	attach();

	return {
		update(nextOptions: StickToBottomOptions) {
			if (nextOptions.root === root) return;
			detach();
			root = nextOptions.root;
			pinned = true;
			attach();
		},
		destroy() {
			detach();
			observer.disconnect();
		}
	};
}
