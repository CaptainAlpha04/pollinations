/**
 * image.pollinations.ai — 301 redirect to gen.pollinations.ai
 * Legacy service has moved. All requests redirect permanently.
 */
const counts: Record<string, number> = {};
let total = 0;

export default {
    async fetch(request: Request): Promise<Response> {
        const url = new URL(request.url);

        if (url.pathname === "/_stats") {
            const top = Object.entries(counts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 50);
            return new Response(
                JSON.stringify({ total, paths: Object.fromEntries(top) }),
                {
                    headers: {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                    },
                },
            );
        }

        const key = "/" + (url.pathname.split("/")[1] || "");
        counts[key] = (counts[key] || 0) + 1;
        total++;

        const newUrl = `https://gen.pollinations.ai${url.pathname}${url.search}`;
        return new Response(
            JSON.stringify({
                message: "This service has moved to gen.pollinations.ai",
                redirect: newUrl,
                status: 301,
            }),
            {
                status: 301,
                headers: {
                    Location: newUrl,
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
            },
        );
    },
};
