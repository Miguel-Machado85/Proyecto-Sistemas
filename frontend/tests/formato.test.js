// Se corre con: node --test frontend/tests/
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { analizarFormato } = require('../assets/formato.js');

const normal = (texto) => ({ texto, negrilla: false });
const fuerte = (texto) => ({ texto, negrilla: true });
const linea = (...segmentos) => ({ tipo: 'linea', segmentos });
const lista = (...items) => ({ tipo: 'lista', items });

test('negrilla con ** y con __', () => {
  assert.deepEqual(analizarFormato('Vence el **15 de marzo** y cuesta __$5.500__.'), [
    linea(normal('Vence el '), fuerte('15 de marzo'), normal(' y cuesta '), fuerte('$5.500'), normal('.')),
  ]);
});

test('lista de varias líneas con -, * y •', () => {
  assert.deepEqual(analizarFormato('Requisitos:\n- Cédula\n  * Recibo\n• Foto'), [
    linea(normal('Requisitos:')),
    lista([normal('Cédula')], [normal('Recibo')], [normal('Foto')]),
  ]);
});

test('lista entre párrafos: los blancos pegados a la lista se descartan, los de párrafos se conservan', () => {
  assert.deepEqual(analizarFormato('Intro\n\n- uno\n- **dos**\n\nCierre\n\nOtro párrafo'), [
    linea(normal('Intro')),
    lista([normal('uno')], [fuerte('dos')]),
    linea(normal('Cierre')),
    { tipo: 'espacio' },
    linea(normal('Otro párrafo')),
  ]);
});

test('encabezado: se quitan los # y la línea queda en negrilla', () => {
  assert.deepEqual(analizarFormato('## Requisitos\nTexto'), [
    linea(fuerte('Requisitos')),
    linea(normal('Texto')),
  ]);
  assert.deepEqual(analizarFormato('# Un **título**'), [linea(fuerte('Un '), fuerte('título'))]);
});

test('cita con corchetes queda literal', () => {
  assert.deepEqual(analizarFormato('Dato [Decreto 001 de 2019, p. 3].'), [linea(normal('Dato [Decreto 001 de 2019, p. 3].'))]);
});

test('** sin cerrar se muestra tal cual y no rompe el resto', () => {
  assert.deepEqual(analizarFormato('Texto **sin cerrar'), [linea(normal('Texto **sin cerrar'))]);
  assert.deepEqual(analizarFormato('Un ** suelto y **otro** bien'), [
    linea(normal('Un ** suelto y '), fuerte('otro'), normal(' bien')),
  ]);
});

test('<script> queda como texto', () => {
  assert.deepEqual(analizarFormato('<script>alert(1)</script>'), [linea(normal('<script>alert(1)</script>'))]);
  assert.deepEqual(analizarFormato('- <img src=x onerror=alert(1)>'), [lista([normal('<img src=x onerror=alert(1)>')])]);
});

test('lista numerada, guiones y asteriscos en medio de la frase quedan como texto normal', () => {
  assert.deepEqual(analizarFormato('1. Pida el turno\n2. Pague - luego regrese a las 5 * 2'), [
    linea(normal('1. Pida el turno')),
    linea(normal('2. Pague - luego regrese a las 5 * 2')),
  ]);
});

test('texto sin formato', () => {
  assert.deepEqual(analizarFormato('Hola.\n\nGracias.'), [
    linea(normal('Hola.')),
    { tipo: 'espacio' },
    linea(normal('Gracias.')),
  ]);
  assert.deepEqual(analizarFormato(''), []);
});

test('formato.js no usa innerHTML, outerHTML ni insertAdjacentHTML', () => {
  const codigo = fs.readFileSync(path.join(__dirname, '../assets/formato.js'), 'utf8')
    .split('\n').filter((l) => !l.trim().startsWith('//')).join('\n');
  assert.doesNotMatch(codigo, /innerHTML|outerHTML|insertAdjacentHTML/);
});
