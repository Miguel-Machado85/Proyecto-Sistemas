// Formato mínimo de las respuestas del agente: **negrilla**, listas con viñetas y encabezados (#) como negrilla.
// Todo lo demás se muestra literal. Sin librerías y sin innerHTML: el DOM se arma con createElement/createTextNode.
//
// analizarFormato(texto) es pura (sin DOM) y devuelve bloques:
//   { tipo: 'linea', segmentos: [{ texto, negrilla }] }
//   { tipo: 'espacio' }                               línea en blanco entre párrafos
//   { tipo: 'lista', items: [[{ texto, negrilla }]] }
// pintarFormato(elemento, texto) pinta esos bloques dentro de `elemento` (sustituye lo que tuviera).

var PATRON_NEGRILLA = /(\*\*|__)(?=\S)(.+?)(?<=\S)\1/g;
var PATRON_VINETA = /^[ \t]*[-*•][ \t]+(\S.*)$/;
var PATRON_ENCABEZADO = /^[ \t]*#{1,3}[ \t]+(\S.*)$/;

function analizarSegmentos(linea, forzarNegrilla) {
  var segmentos = [];
  var desde = 0;
  var coincidencia;
  PATRON_NEGRILLA.lastIndex = 0;
  while ((coincidencia = PATRON_NEGRILLA.exec(linea)) !== null) {
    if (coincidencia.index > desde) {
      segmentos.push({ texto: linea.slice(desde, coincidencia.index), negrilla: !!forzarNegrilla });
    }
    segmentos.push({ texto: coincidencia[2], negrilla: true });
    desde = coincidencia.index + coincidencia[0].length;
  }
  if (desde < linea.length) {
    segmentos.push({ texto: linea.slice(desde), negrilla: !!forzarNegrilla });
  }
  return segmentos;
}

function analizarFormato(texto) {
  var bloques = [];
  var lineas = String(texto == null ? '' : texto).replace(/\r\n?/g, '\n').split('\n');
  var lista = null;

  lineas.forEach(function (linea) {
    var vineta = PATRON_VINETA.exec(linea);
    if (vineta) {
      if (!lista) {
        lista = { tipo: 'lista', items: [] };
        bloques.push(lista);
      }
      lista.items.push(analizarSegmentos(vineta[1], false));
      return;
    }
    lista = null;
    var encabezado = PATRON_ENCABEZADO.exec(linea);
    if (encabezado) {
      bloques.push({ tipo: 'linea', segmentos: analizarSegmentos(encabezado[1], true) });
    } else if (linea.trim() === '') {
      bloques.push({ tipo: 'espacio' });
    } else {
      bloques.push({ tipo: 'linea', segmentos: analizarSegmentos(linea, false) });
    }
  });

  // La lista es un bloque con su propio margen: las líneas en blanco pegadas a ella (y las de los extremos)
  // sobran y con white-space: pre-wrap se verían como huecos de más.
  return bloques.filter(function (bloque, i) {
    if (bloque.tipo !== 'espacio') return true;
    var anterior = bloques[i - 1];
    var siguiente = bloques[i + 1];
    if (!anterior || !siguiente) return false;
    return anterior.tipo !== 'lista' && siguiente.tipo !== 'lista';
  });
}

function pintarSegmentos(padre, segmentos) {
  segmentos.forEach(function (segmento) {
    var nodo = document.createTextNode(segmento.texto);
    if (segmento.negrilla) {
      var fuerte = document.createElement('strong');
      fuerte.appendChild(nodo);
      padre.appendChild(fuerte);
    } else {
      padre.appendChild(nodo);
    }
  });
}

function pintarFormato(elemento, texto) {
  while (elemento.firstChild) elemento.removeChild(elemento.firstChild);
  var anteriorEsTexto = false;
  analizarFormato(texto).forEach(function (bloque) {
    if (bloque.tipo === 'lista') {
      var ul = document.createElement('ul');
      ul.className = 'msg-lista';
      bloque.items.forEach(function (item) {
        var li = document.createElement('li');
        pintarSegmentos(li, item);
        ul.appendChild(li);
      });
      elemento.appendChild(ul);
      anteriorEsTexto = false;
      return;
    }
    // pre-wrap respeta los \n: se ponen solo entre líneas de texto consecutivas.
    if (anteriorEsTexto) elemento.appendChild(document.createTextNode('\n'));
    if (bloque.tipo === 'linea') pintarSegmentos(elemento, bloque.segmentos);
    anteriorEsTexto = true;
  });
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { analizarFormato: analizarFormato, pintarFormato: pintarFormato };
}
