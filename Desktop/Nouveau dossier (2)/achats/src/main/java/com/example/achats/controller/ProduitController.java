package com.example.achats.controller;

import com.example.achats.model.Produit;
import com.example.achats.service.ProduitService;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * Contrôleur REST API bech naamlou gestion des produits.
 */
@RestController
@RequestMapping("/api/produits")
public class ProduitController {
    
    private final ProduitService produitService;

    /**
     *  le service de gestion des produits
     */

    public ProduitController(ProduitService produitService) {
        this.produitService = produitService;
    }

    /**
     * ta3mel select lel produits lkol
     */
    @GetMapping
    public ResponseEntity<Page<Produit>> getAllProduits(Pageable pageable) {
        return ResponseEntity.ok(produitService.getAllProduits(pageable));
    }

    /**
     * ta3mel select l produit bel id mte3ou
     */
    @GetMapping("/{id}")
    public ResponseEntity<Produit> getProduitById(@PathVariable Long id) {
        return produitService.getProduitById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    /**
     tasna3 produits jdid
     */
    @PostMapping
    public ResponseEntity<Produit> createProduit(@RequestBody Produit produit) {
        return ResponseEntity.ok(produitService.createProduit(produit));
    }

    /**
     ta3mel edit
     */
    @PutMapping("/{id}")
    public ResponseEntity<Produit> updateProduit(@PathVariable Long id, @RequestBody Produit produitDetails) {
        return ResponseEntity.ok(produitService.updateProduit(id, produitDetails));
    }

    /**ta3mel supprimer
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteProduit(@PathVariable Long id) {
        produitService.deleteProduit(id);
        return ResponseEntity.noContent().build();
    }

    /**
    ta3mel filtrage par categorie(bien waela services
     */
    @GetMapping("/categorie/{categorieId}")
    public ResponseEntity<Page<Produit>> getProduitsByCategorie(
            @PathVariable Long categorieId,
            Pageable pageable) {
        return ResponseEntity.ok(produitService.getProduitsByCategorie(categorieId, pageable));
    }
}
