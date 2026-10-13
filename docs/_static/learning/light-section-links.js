(() => {
  'use strict';
  if (!location.pathname.endsWith('/learn/fundamentals/light.html')) return;
  const destinations = {
  "multiple-incident-waves": "multiple-beams.html#multiple-incident-waves",
  "id10": "multiple-beams.html#multiple-incident-waves",
  "excitation-amplitude-phase": "multiple-beams.html#excitation-amplitude-phase",
  "id12": "multiple-beams.html#excitation-amplitude-phase",
  "same-side-beams": "multiple-beams.html#same-side-beams",
  "id13": "multiple-beams.html#beams-detail-id13",
  "id30": "multiple-beams.html#beams-detail-id30",
  "id14": "multiple-beams.html#beams-detail-id30",
  "id29": "multiple-beams.html#beams-detail-id29",
  "id15": "multiple-beams.html#beams-detail-id15",
  "two-sided-illumination": "multiple-beams.html#two-sided-illumination",
  "id16": "multiple-beams.html#two-sided-illumination",
  "id19": "multiple-beams.html#beams-detail-id19",
  "id17": "multiple-beams.html#beams-detail-id17",
  "id20": "multiple-beams.html#beams-detail-id20",
  "id18": "multiple-beams.html#beams-detail-id20",
  "beam": "multiple-beams.html#beam-input-support",
  "beam-input-support": "multiple-beams.html#beam-input-support",
  "incident-order-inputs": "multiple-beams.html#incident-order-inputs",
  "id21": "multiple-beams.html#incident-order-inputs",
  "id22": "multiple-beams.html#beams-detail-id22",
  "id23": "multiple-beams.html#beams-detail-id22",
  "id24": "multiple-beams.html#incident-order-inputs",
  "id25": "multiple-beams.html#incident-order-inputs",
  "id26": "multiple-beams.html#incident-order-inputs",
  "id27": "multiple-beams.html#incident-order-inputs",
  "d-m-n": "multiple-beams.html#d-m-n",
  "excitation-order-pairs": "multiple-beams.html#excitation-order-pairs",
  "id28": "multiple-beams.html#d-m-n",
  "id31": "multiple-beams.html#beams-detail-id31",
  "id32": "multiple-beams.html#beams-detail-id31",
  "coherent-and-incoherent": "multiple-beams.html#coherent-and-incoherent",
  "id33": "multiple-beams.html#coherent-and-incoherent",
  "coherent-illumination-flow": "multiple-beams.html#coherent-illumination-flow",
  "id34": "multiple-beams.html#coherent-illumination-flow",
  "id56": "multiple-beams.html#coherent-illumination-flow",
  "id35": "multiple-beams.html#illumination-power-region",
  "illumination-power-region": "multiple-beams.html#illumination-power-region",
  "id36": "multiple-beams.html#illumination-power-region",
  "id37": "fields.html#illumination-fields",
  "illumination-fields": "fields.html#illumination-fields",
  "id38": "fields.html#illumination-fields",
  "id39": "fields.html#illumination-fields",
  "id40": "fields.html#illumination-fields",
  "id41": "fields.html#illumination-fields",
  "id42": "fields.html#illumination-fields",
  "id43": "fields.html#illumination-fields",
  "id44": "fields.html#illumination-fields",
  "magnetic-field-units": "fields.html#magnetic-field-units",
  "id45": "fields.html#magnetic-field-units",
  "id11": "light.html#light-detail-id11",
  "id4": "light.html#light-detail-id4",
  "id7": "light.html#light-detail-id7"
};
  const followSection = () => {
    const target = destinations[location.hash.slice(1)];
    if (target) location.replace(new URL(target, location.href).href);
  };
  followSection();
  window.addEventListener('hashchange', followSection);
})();
